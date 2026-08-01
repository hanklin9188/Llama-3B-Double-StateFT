import os

import torch
from transformers import Trainer

from .io.state import load_control_state_if_any, save_control_state


class AdapterOnlyTrainer(Trainer):
    """Trainer that checkpoints only Double Control weights, not the frozen base model."""

    def __init__(self, *args, control_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.control_config = control_config

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        save_control_state(self.model, output_dir, self.control_config)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None, **kwargs):
        target = model or self.model
        if load_control_state_if_any(target, resume_from_checkpoint) is not None:
            raise FileNotFoundError(
                f"No control_state.pt in checkpoint: {resume_from_checkpoint}"
            )

    def _load_best_model(self):
        if not self.state.best_model_checkpoint:
            return
        if load_control_state_if_any(self.model, self.state.best_model_checkpoint) is not None:
            raise FileNotFoundError(
                f"Best checkpoint has no control_state.pt: {self.state.best_model_checkpoint}"
            )
