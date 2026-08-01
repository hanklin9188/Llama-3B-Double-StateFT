from ..common_imports import *
from ..adapters.control import ControlLoRA


def iter_control_modules(module: nn.Module):
    for m in module.modules():
        if isinstance(m, ControlLoRA):
            yield m


def save_control_state(model: nn.Module, out_dir: str, cfg: Dict[str, Any]):
    os.makedirs(out_dir, exist_ok=True)
    state = {f"ctrl_{i}": m.state_dict() for i, m in enumerate(iter_control_modules(model))}
    torch.save(state, os.path.join(out_dir, "control_state.pt"))
    import json
    with open(os.path.join(out_dir, "control_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def load_control_state_if_any(model: nn.Module, maybe_dir: Optional[str]) -> Optional[str]:
    if not maybe_dir:
        return None
    pt = os.path.join(maybe_dir, "control_state.pt")
    if os.path.exists(pt):
        state = torch.load(pt, map_location="cpu")
        modules = list(iter_control_modules(model))
        if len(state) != len(modules):
            raise ValueError(
                f"Control module count mismatch: checkpoint={len(state)}, model={len(modules)}"
            )
        for i, m in enumerate(modules):
            key = f"ctrl_{i}"
            if key not in state:
                raise KeyError(f"Missing {key} in {pt}")
            m.load_state_dict(state[key], strict=True)
        print(f"[Resume] Loaded control_state from {pt}")
        return None
    return maybe_dir


class ControlSaveCallback(TrainerCallback):
    def __init__(self, model, cfg):
        self.model = model
        self.cfg = cfg
    def on_save(self, args, state, control, **kwargs):
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt_dir):
            save_control_state(self.model, ckpt_dir, self.cfg)
            print(f"[ControlSaveCallback] saved control_state.pt into {ckpt_dir}")
