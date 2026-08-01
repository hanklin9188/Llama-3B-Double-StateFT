from ..common_imports import *
def normalize_repo_id(repo: str) -> str:
    repo = str(repo).strip().strip("'\"")
    if os.path.isdir(repo):
        return repo
    repo = repo.replace("\\", "/")
    repo = "/".join([p for p in repo.split("/") if p])
    parts = repo.split("/")
    if len(parts) >= 3 and parts[0].lower() == parts[1].lower():
        parts = [parts[0]] + parts[2:]
        repo = "/".join(parts)
    aliases = {
        "llama-3.2-1b": "meta-llama/Llama-3.2-1B",
        "llama-3.2-3b": "meta-llama/Llama-3.2-3B",
    }
    return aliases.get(repo.lower(), repo)


def require_namespace_format(repo: str):
    if os.path.isdir(repo):
        return
    if "/" not in repo:
        raise ValueError(
            f"[Invalid --base_model] '{repo}' 不是合法的 HuggingFace repo id。\n"
            f"請用 'namespace/repo_name' 格式，例如：meta-llama/Llama-3.2-3B"
        )
