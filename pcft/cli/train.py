import gc
import os

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from ..data import tokenize_example
from ..io.state import export_compact_adapter, iter_control_modules, verify_compact_adapter
from ..trainer import BranchGradientEMACallback, DynamicRankCallback, IDDRTrainer
from ..utils.repo import normalize_repo_id, require_namespace_format
from ..wrapping.wrap import _find_llama_like_layers_container, wrap_model_with_double_control


def _split_dataset(raw, rank_calibration_size, validation_size, seed):
    holdout_size = int(rank_calibration_size) + int(validation_size)
    if holdout_size <= 0 or holdout_size >= len(raw):
        raise ValueError("rank_calibration_size + validation_size must be between 1 and dataset size")
    split = raw.train_test_split(test_size=holdout_size, seed=seed, shuffle=True)
    holdout = split["test"].train_test_split(
        test_size=validation_size,
        train_size=rank_calibration_size,
        seed=seed,
        shuffle=True,
    )
    return split["train"], holdout["train"], holdout["test"]


def train_id_dr_stateft(
    base_model="meta-llama/Llama-3.2-3B",
    data_path="datasets/commonsense_170k.json",
    output_dir="checkpoints/llama-3.2-3b-id-dr-stateft",
    allocation_method="id_exchange",
    initial_rank=64,
    rank_min=8,
    rank_max=128,
    rank_quantum=8,
    global_rank_budget=None,
    alpha=16.0,
    scaling_mode="alpha_over_sqrt_rank",
    dropout=0.05,
    batch_size=16,
    micro_batch_size=4,
    num_epochs=3,
    max_steps=-1,
    learning_rate=5e-5,
    cutoff_len=256,
    rank_calibration_size=1000,
    validation_size=2000,
    eval_steps=200,
    save_steps=200,
    allocation_interval=600,
    allocation_warmup_events=3,
    rank_freeze_ratio=0.2,
    id_sample_size=256,
    id_bootstrap_repeats=20,
    id_bootstrap_fraction=0.8,
    id_uncertainty_lambda=1.0,
    id_ema_alpha=0.3,
    id_prior_beta=2.0,
    id_regularization_tau=0.05,
    rank_switch_cost=0.001,
    move_threshold=0.0,
    rank_probe_size=128,
    direct_verify_size=256,
    receiver_count=10,
    donor_count=10,
    direct_verify_pairs=6,
    max_transfers_per_event=4,
    rank_exploration_probability=0.15,
    exploration_transfers=1,
    rank_transition_steps=0,
    train_on_inputs=True,
    resume_from_checkpoint=None,
    seed=42,
):
    from datasets import load_dataset

    if "8b" in base_model.lower():
        raise ValueError("This project supports Llama 3B only")
    if allocation_method not in {"fixed", "loss_exchange", "id_exchange"}:
        raise ValueError("allocation_method must be fixed, loss_exchange, or id_exchange")
    if not (rank_min <= initial_rank <= rank_max):
        raise ValueError("Expected rank_min <= initial_rank <= rank_max")
    if any(rank % rank_quantum for rank in (rank_min, initial_rank, rank_max)):
        raise ValueError("All ranks must be divisible by rank_quantum")

    set_seed(seed)
    base_model = normalize_repo_id(base_model)
    require_namespace_format(base_model)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
    model_kwargs = {"torch_dtype": dtype, "device_map": "auto"}
    tokenizer_kwargs = {"use_fast": True}
    if token:
        model_kwargs["token"] = token
        tokenizer_kwargs["token"] = token
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(base_model, **tokenizer_kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.config.use_cache = False

    _, _, layers, _ = _find_llama_like_layers_container(model)
    layer_count = len(layers)
    branch_count = 2 * layer_count
    global_rank_budget = int(global_rank_budget or branch_count * initial_rank)
    if global_rank_budget != branch_count * initial_rank:
        raise ValueError("Initial uniform rank must exactly match global_rank_budget")
    model = wrap_model_with_double_control(
        model,
        ranks_attn=[initial_rank] * layer_count,
        ranks_mlp=[initial_rank] * layer_count,
        rank_max_attn=[rank_max] * layer_count,
        rank_max_mlp=[rank_max] * layer_count,
        alpha=alpha,
        dropout=dropout,
        rank_quantum=rank_quantum,
        scaling_mode=scaling_mode,
        # TrainingArguments checkpoints the whole wrapped decoder layer.
        checkpoint_base=False,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for control in iter_control_modules(model):
        for parameter in control.parameters():
            parameter.requires_grad_(True)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    active_parameters = 2 * model.config.hidden_size * global_rank_budget
    print(
        f"[ID-DR] layers={layer_count}, branches={branch_count}, "
        f"supernet_params={trainable:,}, active_params={active_parameters:,}, "
        f"global_budget={global_rank_budget}"
    )

    raw = load_dataset("json", data_files=data_path)["train"]
    train_raw, rank_raw, validation_raw = _split_dataset(
        raw, rank_calibration_size, validation_size, seed
    )
    columns = raw.column_names
    mapping = lambda row: tokenize_example(row, tokenizer, cutoff_len, train_on_inputs)
    train_data = train_raw.shuffle(seed=seed).map(mapping, remove_columns=columns)
    rank_data = rank_raw.map(mapping, remove_columns=columns)
    validation_data = validation_raw.map(mapping, remove_columns=columns)
    print(
        f"[Data] train={len(train_data)}, rank_calibration={len(rank_data)}, "
        f"validation={len(validation_data)}"
    )

    gradient_accumulation = max(1, batch_size // micro_batch_size)
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="steps",
        save_steps=save_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
        save_safetensors=False,
    )
    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, padding=True, pad_to_multiple_of=8, return_tensors="pt"
    )
    control_config = {
        "architecture": "id_dr_double_control_v2",
        "base_model": base_model,
        "hidden_size": model.config.hidden_size,
        "rank_min": rank_min,
        "rank_max": rank_max,
        "rank_quantum": rank_quantum,
        "initial_rank": initial_rank,
        "global_rank_budget": global_rank_budget,
        "alpha": alpha,
        "dropout": dropout,
        "scaling_mode": scaling_mode,
        "compact": False,
        "allocation_method": allocation_method,
        "allocator": {
            "id_prior_beta": id_prior_beta,
            "id_regularization_tau": id_regularization_tau,
            "rank_switch_cost": rank_switch_cost,
            "move_threshold": move_threshold,
            "rank_exploration_probability": rank_exploration_probability,
            "rank_freeze_ratio": rank_freeze_ratio,
            "allocation_interval": allocation_interval,
            "allocation_warmup_events": allocation_warmup_events,
            "receiver_count": receiver_count,
            "donor_count": donor_count,
            "direct_verify_pairs": direct_verify_pairs,
            "max_transfers_per_event": max_transfers_per_event,
        },
        "rank_budget_mode": "global_double",
        "nested_rank": {
            "exploration_probability": rank_exploration_probability,
            "exploration_transfers": exploration_transfers,
            "transition_steps": rank_transition_steps,
        },
        "geometry": {
            "sample_size": id_sample_size,
            "bootstrap_repeats": id_bootstrap_repeats,
            "bootstrap_fraction": id_bootstrap_fraction,
            "uncertainty_lambda": id_uncertainty_lambda,
            "id_ema_alpha": id_ema_alpha,
        },
        "rank_probe": {
            "probe_examples": rank_probe_size,
            "direct_verify_examples": direct_verify_size,
        },
    }
    trainer = IDDRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        rank_calibration_dataset=rank_data,
        data_collator=collator,
        processing_class=tokenizer,
        control_config=control_config,
        exploration_probability=(0.0 if allocation_method == "fixed" else rank_exploration_probability),
        exploration_transfers=exploration_transfers,
    )
    if allocation_method != "fixed":
        gradient_callback = BranchGradientEMACallback()
        trainer.gradient_callback = gradient_callback
        trainer.add_callback(gradient_callback)
        dynamic_callback = DynamicRankCallback(
            trainer,
            allocation_interval=allocation_interval,
            warmup_events=allocation_warmup_events,
            freeze_ratio=rank_freeze_ratio,
            id_sample_size=id_sample_size,
            bootstrap_repeats=id_bootstrap_repeats,
            bootstrap_fraction=id_bootstrap_fraction,
            uncertainty_lambda=id_uncertainty_lambda,
            probe_examples=rank_probe_size,
            direct_examples=direct_verify_size,
            transition_steps=rank_transition_steps,
            rank_min=rank_min,
            rank_max=rank_max,
            rank_quantum=rank_quantum,
            global_budget=global_rank_budget,
            id_prior_beta=id_prior_beta,
            id_regularization_tau=id_regularization_tau,
            switch_cost=rank_switch_cost,
            move_threshold=move_threshold,
            receiver_count=receiver_count,
            donor_count=donor_count,
            direct_verify_pairs=direct_verify_pairs,
            max_transfers=max_transfers_per_event,
            id_ema_alpha=id_ema_alpha,
            use_id_prior=allocation_method == "id_exchange",
        )
        trainer.dynamic_callback = dynamic_callback
        trainer.add_callback(dynamic_callback)

    latest = None
    if os.path.isdir(output_dir):
        numeric = [
            name for name in os.listdir(output_dir)
            if name.startswith("checkpoint-") and name.split("-")[-1].isdigit()
        ]
        if numeric:
            latest = os.path.join(output_dir, max(numeric, key=lambda name: int(name.split("-")[-1])))
    trainer.train(resume_from_checkpoint=resume_from_checkpoint or latest)
    trainer.save_model(output_dir)
    compact_dir = export_compact_adapter(output_dir, os.path.join(output_dir, "compact"))
    compact_error = verify_compact_adapter(output_dir, compact_dir)
    print(f"[Compact] verified max branch error={compact_error:.3e}")
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"adapter_dir": output_dir, "compact_dir": str(compact_dir)}


# Compatibility alias for older scripts.
train_double_control = train_id_dr_stateft
