import gc
import os

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from ..data import tokenize_example
from ..io.state import iter_control_modules, save_control_state
from ..metrics.adaptive_rank import AdaptiveRankAllocatorCallback
from ..metrics.callbacks_basic import EvalLossRecorderCallback, GradNormRecorderCallback
from ..metrics.callbacks_idii import EvalIDIIRecorderCallback
from ..trainer import AdapterOnlyTrainer
from ..utils.repo import normalize_repo_id, require_namespace_format
from ..wrapping.wrap import _find_llama_like_layers_container, wrap_model_with_double_control


def train_double_control(
    base_model="meta-llama/Llama-3.2-3B",
    data_path="datasets/commonsense_170k.json",
    output_dir="checkpoints/llama-3.2-3b-double",
    initial_rank=64,
    rank_min=8,
    rank_max=128,
    alpha=16.0,
    dropout=0.05,
    batch_size=16,
    micro_batch_size=4,
    num_epochs=3,
    learning_rate=5e-5,
    cutoff_len=256,
    val_set_size=2000,
    eval_steps=200,
    save_steps=200,
    id_sample_size=256,
    warmup_evals=3,
    resume_from_checkpoint=None,
    seed=42,
    train_on_inputs=True,
):
    from datasets import load_dataset

    if "8b" in base_model.lower():
        raise ValueError("This project supports Llama 3B only")
    if not (1 <= rank_min <= initial_rank <= rank_max):
        raise ValueError("Expected rank_min <= initial_rank <= rank_max")

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
    module_ranks = [rank_max] * layer_count
    alphas = [alpha] * layer_count
    model = wrap_model_with_double_control(
        model,
        ranks=module_ranks,
        alphas=alphas,
        dropout=dropout,
        checkpoint_base=True,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for control in iter_control_modules(model):
        control.set_active_rank(initial_rank)
        for parameter in control.parameters():
            parameter.requires_grad_(True)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    print(f"[DoubleControl] layers={layer_count}, trainable={trainable:,}")
    print(
        f"[DoubleControl] rank min/initial/max={rank_min}/{initial_rank}/{rank_max}; "
        f"per-branch budget={layer_count * initial_rank}"
    )

    raw = load_dataset("json", data_files=data_path)["train"]
    validation_size = min(int(val_set_size), max(0, len(raw) - 1))
    if validation_size:
        split = raw.train_test_split(test_size=validation_size, seed=seed, shuffle=True)
        train_raw, validation_raw = split["train"], split["test"]
    else:
        train_raw, validation_raw = raw, None
    columns = raw.column_names
    mapping = lambda row: tokenize_example(row, tokenizer, cutoff_len, train_on_inputs)
    train_data = train_raw.shuffle(seed=seed).map(mapping, remove_columns=columns)
    validation_data = (
        validation_raw.shuffle(seed=seed).map(mapping, remove_columns=columns)
        if validation_raw is not None
        else None
    )

    gradient_accumulation = max(1, batch_size // micro_batch_size)
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="steps",
        save_steps=save_steps,
        eval_strategy="steps" if validation_data is not None else "no",
        eval_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=validation_data is not None,
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
        tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    control_config = {
        "architecture": "double_control",
        "base_model": base_model,
        "module_ranks": module_ranks,
        "alphas": alphas,
        "dropout": dropout,
        "initial_rank": initial_rank,
        "rank_min": rank_min,
        "rank_max": rank_max,
        "budget_semantics": "fixed_per_branch",
    }
    trainer = AdapterOnlyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=collator,
        processing_class=tokenizer,
        control_config=control_config,
    )
    if validation_data is not None:
        metrics_dir = os.path.join(output_dir, "metrics")
        trainer.add_callback(
            EvalIDIIRecorderCallback(
                trainer,
                sample_size=id_sample_size,
                results_dir=metrics_dir,
                control_cfg=control_config,
            )
        )
        gradient_callback = GradNormRecorderCallback(beta=0.9)
        loss_callback = EvalLossRecorderCallback(window=5)
        trainer.add_callback(gradient_callback)
        trainer.add_callback(loss_callback)
        trainer.add_callback(
            AdaptiveRankAllocatorCallback(
                results_dir=metrics_dir,
                total_budget=layer_count * initial_rank,
                rank_min=rank_min,
                rank_max=rank_max,
                grad_recorder=gradient_callback,
                loss_recorder=loss_callback,
                warmup_evals=warmup_evals,
            )
        )

    latest = None
    if os.path.isdir(output_dir):
        checkpoints = [name for name in os.listdir(output_dir) if name.startswith("checkpoint-")]
        numeric = [name for name in checkpoints if name.split("-")[-1].isdigit()]
        if numeric:
            latest = os.path.join(output_dir, max(numeric, key=lambda name: int(name.split("-")[-1])))
    resume = resume_from_checkpoint or latest
    trainer.train(resume_from_checkpoint=resume)

    os.makedirs(output_dir, exist_ok=True)
    save_control_state(model, output_dir, control_config)
    tokenizer.save_pretrained(output_dir)
    del trainer, model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_dir
