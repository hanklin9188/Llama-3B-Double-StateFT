import csv
import json
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from .io.state import load_control_state_if_any
from .wrapping.wrap import wrap_model_with_double_control


DATASETS = [
    "boolq",
    "piqa",
    "social_i_qa",
    "hellaswag",
    "winogrande",
    "ARC-Challenge",
    "ARC-Easy",
    "openbookqa",
]


def _candidates(dataset):
    if dataset == "boolq":
        return ["true", "false"]
    if dataset == "piqa":
        return ["solution1", "solution2"]
    if dataset == "winogrande":
        return ["option1", "option2"]
    if dataset == "hellaswag":
        return ["ending1", "ending2", "ending3", "ending4"]
    if dataset == "social_i_qa":
        return ["answer1", "answer2", "answer3"]
    return ["answer1", "answer2", "answer3", "answer4"]


def _hint(dataset):
    return "Answer with exactly one of: " + ", ".join(_candidates(dataset)) + "."


def _prompt(example, dataset):
    instruction = example.get("instruction") or example.get("question") or ""
    context = example.get("input") or example.get("context")
    input_section = f"\n\n### Input:\n{context}" if context else ""
    return (
        "Below is an instruction that describes a task. Write a response that appropriately "
        "completes the request.\n\n"
        f"### Instruction:\n{instruction}{input_section}\n\n{_hint(dataset)}\n\n"
        "### Response:\nAnswer:"
    )


def _input_device(model):
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for name, device in device_map.items():
            if "embed_tokens" in name and str(device) not in {"cpu", "disk", "meta"}:
                return torch.device(device)
    return next(model.parameters()).device


@torch.no_grad()
def _score_candidates(model, tokenizer, prompt, candidates, length_norm="none"):
    device = _input_device(model)
    prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0].to(device)
    scores = {}
    for candidate in candidates:
        candidate_ids = tokenizer.encode(" " + candidate, add_special_tokens=False)
        candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=device)
        input_ids = torch.cat([prompt_ids, candidate_tensor])[None, :]
        logits = model(input_ids=input_ids, use_cache=False, return_dict=True).logits.float()
        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        start = len(prompt_ids) - 1
        token_scores = log_probs[0, start : start + len(candidate_ids)].gather(
            1, candidate_tensor[:, None]
        )
        score = token_scores.sum().item()
        if length_norm == "avg":
            score /= max(1, len(candidate_ids))
        scores[candidate] = score
    return max(scores, key=scores.get)


def read_adapter_config(adapter_dir):
    adapter_dir = Path(adapter_dir)
    config_path = adapter_dir / "control_config.json"
    weights_path = adapter_dir / "control_state.pt"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Incomplete ID-DR adapter: {adapter_dir}")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("architecture") != "id_dr_double_control_v2":
        raise ValueError("Expected an id_dr_double_control_v2 checkpoint")
    if not config.get("layers"):
        raise ValueError("Adapter config has no per-branch layer ranks")
    return config


def load_adapter(base_model, adapter_dir):
    config = read_adapter_config(adapter_dir)
    base_model = base_model or config["base_model"]
    layers = sorted(config["layers"], key=lambda row: int(row["layer"]))
    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    kwargs = {"torch_dtype": dtype, "device_map": "auto"}
    if token:
        kwargs["token"] = token
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    model = wrap_model_with_double_control(
        model,
        ranks_attn=[int(row["rank_attn"]) for row in layers],
        ranks_mlp=[int(row["rank_mlp"]) for row in layers],
        rank_max_attn=[int(row["rank_max_attn"]) for row in layers],
        rank_max_mlp=[int(row["rank_max_mlp"]) for row in layers],
        alpha=float(config["alpha"]),
        dropout=0.0,
        rank_quantum=int(config["rank_quantum"]),
        scaling_mode=config["scaling_mode"],
        checkpoint_base=False,
    )
    load_control_state_if_any(model, str(adapter_dir))
    tokenizer = AutoTokenizer.from_pretrained(base_model, token=token, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def evaluate_adapter(
    base_model,
    adapter_dir,
    test_data_path,
    datasets=None,
    results_file=None,
    length_norm="none",
    max_examples=None,
):
    datasets = datasets or DATASETS
    adapter_config = read_adapter_config(adapter_dir)
    adapter_parameters = int(
        adapter_config.get(
            "configured_parameters",
            2
            * int(adapter_config["hidden_size"])
            * sum(
                int(row["rank_attn"]) + int(row["rank_mlp"])
                for row in adapter_config["layers"]
            ),
        )
    )
    model, tokenizer = load_adapter(base_model, adapter_dir)
    rows = []
    for dataset in datasets:
        test_path = Path(test_data_path) / dataset / "test.json"
        if not test_path.is_file():
            raise FileNotFoundError(f"Evaluation data not found: {test_path}")
        with test_path.open(encoding="utf-8") as handle:
            examples = json.load(handle)
        if max_examples is not None:
            examples = examples[: int(max_examples)]
        candidates = _candidates(dataset)
        correct = 0
        elapsed = 0.0
        for example in tqdm(examples, desc=dataset):
            started = time.perf_counter()
            prediction = _score_candidates(
                model, tokenizer, _prompt(example, dataset), candidates, length_norm
            )
            elapsed += time.perf_counter() - started
            answer = example.get("answer", example.get("label"))
            if isinstance(answer, int):
                answer = candidates[answer]
            correct += prediction == answer
        accuracy = correct / max(1, len(examples))
        rows.append(
            {
                "dataset": dataset,
                "accuracy": accuracy,
                "num_examples": len(examples),
                "seconds": elapsed,
                "examples_per_second": len(examples) / max(elapsed, 1e-12),
                "adapter_parameters": adapter_parameters,
            }
        )
        print(f"[Evaluation] {dataset}: {accuracy:.6f}")

    macro = sum(row["accuracy"] for row in rows) / max(1, len(rows))
    rows.append(
        {
            "dataset": "macro_average",
            "accuracy": macro,
            "num_examples": sum(row["num_examples"] for row in rows),
            "seconds": sum(row["seconds"] for row in rows),
            "examples_per_second": 0.0,
            "adapter_parameters": adapter_parameters,
        }
    )
    rows[-1]["examples_per_second"] = rows[-1]["num_examples"] / max(rows[-1]["seconds"], 1e-12)
    output = Path(results_file or Path(adapter_dir) / "evaluation.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Evaluation] macro average: {macro:.6f}")
    print(f"[Evaluation] adapter parameters: {adapter_parameters:,}")
    print(f"[Evaluation] summary: {output}")
    return rows
