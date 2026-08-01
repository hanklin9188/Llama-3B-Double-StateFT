def generate_prompt(example):
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input that "
            "provides further context. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    return (
        "Below is an instruction that describes a task. Write a response that appropriately "
        "completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
    )


def tokenize_example(example, tokenizer, cutoff_len, train_on_inputs=True):
    full_prompt = generate_prompt(example)
    tokenized = tokenizer(
        full_prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
    )
    if (
        tokenized["input_ids"]
        and tokenized["input_ids"][-1] != tokenizer.eos_token_id
        and len(tokenized["input_ids"]) < cutoff_len
    ):
        tokenized["input_ids"].append(tokenizer.eos_token_id)
        tokenized["attention_mask"].append(1)
    tokenized["labels"] = tokenized["input_ids"].copy()

    if not train_on_inputs:
        user_example = {**example, "output": ""}
        user_tokens = tokenizer(
            generate_prompt(user_example),
            truncation=True,
            max_length=cutoff_len,
            padding=False,
        )
        user_length = len(user_tokens["input_ids"])
        tokenized["labels"] = [-100] * user_length + tokenized["labels"][user_length:]
    return tokenized
