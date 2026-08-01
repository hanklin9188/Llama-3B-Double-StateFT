import csv
import json
import os
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


def load_adapter(base_model, adapter_dir):
    adapter_dir = Path(adapter_dir)
    config_path = adapter_dir / "control_config.json"
    weights_path = adapter_dir / "control_state.pt"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Incomplete Double Control adapter: {adapter_dir}")
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("architecture") != "double_control":
        raise ValueError("Adapter is not a Double Control checkpoint")

    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    kwargs = {"torch_dtype": dtype, "device_map": "auto"}
    if token:
        kwargs["token"] = token
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)
    model = wrap_model_with_double_control(
        model,
        ranks=config["module_ranks"],
        alphas=config["alphas"],
        dropout=0.0,
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
):
    datasets = datasets or DATASETS
    model, tokenizer = load_adapter(base_model, adapter_dir)
    rows = []
    for dataset in datasets:
        test_path = Path(test_data_path) / dataset / "test.json"
        with test_path.open(encoding="utf-8") as handle:
            examples = json.load(handle)
        candidates = _candidates(dataset)
        correct = 0
        for example in tqdm(examples, desc=dataset):
            prediction = _score_candidates(
                model,
                tokenizer,
                _prompt(example, dataset),
                candidates,
                length_norm,
            )
            answer = example.get("answer", example.get("label"))
            if isinstance(answer, int):
                answer = candidates[answer]
            correct += prediction == answer
        accuracy = correct / max(1, len(examples))
        rows.append({"dataset": dataset, "accuracy": accuracy})
        print(f"[Evaluation] {dataset}: {accuracy:.6f}")

    output = Path(results_file or Path(adapter_dir) / "evaluation.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Evaluation] summary: {output}")
    return rows
