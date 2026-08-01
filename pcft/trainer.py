import csv
import json
import os
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback

from .io.state import load_control_state_if_any, save_control_state
from .metrics.adaptive_rank import IDRegularizedRankExchangeOptimizer
from .metrics.branch_geometry import append_geometry_csv, measure_branch_geometry
from .rank import (
    BranchKey,
    get_rank_map,
    iter_branch_controls,
    override_rank_map,
    set_rank_map,
    validate_rank_map,
)


class BranchGradientEMACallback(TrainerCallback):
    """Track branch-local gradient energy without defining allocation utility."""

    def __init__(self, beta=0.9):
        self.beta = float(beta)
        self.gradient_ema = {}

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        for key, branch in iter_branch_controls(model):
            squared = 0.0
            for parameter in (branch.lora_A, branch.lora_B):
                if parameter.grad is not None:
                    squared += float(parameter.grad.detach().float().square().sum().item())
            energy = squared ** 0.5
            previous = self.gradient_ema.get(key, energy)
            self.gradient_ema[key] = self.beta * previous + (1.0 - self.beta) * energy


class IDDRTrainer(Trainer):
    def __init__(
        self,
        *args,
        control_config,
        rank_calibration_dataset,
        exploration_probability=0.15,
        exploration_transfers=1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.control_config = control_config
        self.rank_calibration_dataset = rank_calibration_dataset
        self.exploration_probability = float(exploration_probability)
        self.exploration_transfers = int(exploration_transfers)
        self.rank_frozen = False
        self.dynamic_callback = None
        self.gradient_callback = None

    def get_rank_calibration_dataloader(self):
        return DataLoader(
            self.rank_calibration_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
        )

    def _exploration_map(self, model):
        rank_map = get_rank_map(model)
        config = self.control_config
        quantum = int(config["rank_quantum"])
        rank_min = int(config["rank_min"])
        rank_max = int(config["rank_max"])
        for _ in range(self.exploration_transfers):
            receivers = [key for key, rank in rank_map.items() if rank + quantum <= rank_max]
            donors = [key for key, rank in rank_map.items() if rank - quantum >= rank_min]
            if not receivers or not donors:
                break
            receiver = random.choice(receivers)
            donor_options = [key for key in donors if key != receiver]
            if not donor_options:
                break
            donor = random.choice(donor_options)
            rank_map[receiver] += quantum
            rank_map[donor] -= quantum
        validate_rank_map(
            rank_map, rank_min, rank_max, quantum, int(config["global_rank_budget"])
        )
        return rank_map

    def training_step(self, model, inputs, num_items_in_batch=None):
        explore = (
            not self.rank_frozen
            and self.exploration_probability > 0
            and random.random() < self.exploration_probability
        )
        if not explore:
            return super().training_step(model, inputs, num_items_in_batch)
        with override_rank_map(model, self._exploration_map(model)):
            return super().training_step(model, inputs, num_items_in_batch)

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        save_control_state(self.model, output_dir, self.control_config)
        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        if self.dynamic_callback is not None:
            self.dynamic_callback.save_state(output_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None, **kwargs):
        if load_control_state_if_any(model or self.model, resume_from_checkpoint) is not None:
            raise FileNotFoundError(f"No control_state.pt in {resume_from_checkpoint}")
        if self.dynamic_callback is not None:
            self.dynamic_callback.load_state(resume_from_checkpoint)

    def _load_best_model(self):
        if self.state.best_model_checkpoint:
            load_control_state_if_any(self.model, self.state.best_model_checkpoint)


class DynamicRankCallback(TrainerCallback):
    def __init__(
        self,
        trainer: IDDRTrainer,
        allocation_interval=600,
        warmup_events=3,
        freeze_ratio=0.2,
        id_sample_size=256,
        bootstrap_repeats=20,
        bootstrap_fraction=0.8,
        uncertainty_lambda=1.0,
        probe_examples=128,
        direct_examples=256,
        transition_steps=0,
        **optimizer_kwargs,
    ):
        self.trainer = trainer
        self.allocation_interval = int(allocation_interval)
        self.warmup_events = int(warmup_events)
        self.freeze_ratio = float(freeze_ratio)
        self.id_sample_size = int(id_sample_size)
        self.bootstrap_repeats = int(bootstrap_repeats)
        self.bootstrap_fraction = float(bootstrap_fraction)
        self.uncertainty_lambda = float(uncertainty_lambda)
        self.probe_examples = int(probe_examples)
        self.direct_examples = int(direct_examples)
        self.transition_steps = int(transition_steps)
        self.event_count = 0
        self.optimizer = IDRegularizedRankExchangeOptimizer(**optimizer_kwargs)
        self.metrics_dir = Path(trainer.args.output_dir) / "metrics"

    def state_dict(self):
        return {
            "event_count": self.event_count,
            "id_ema": {key.name: value for key, value in self.optimizer.id_ema.items()},
            "gradient_ema": {
                key.name: value
                for key, value in (
                    self.trainer.gradient_callback.gradient_ema.items()
                    if self.trainer.gradient_callback is not None
                    else []
                )
            },
        }

    def save_state(self, directory):
        with (Path(directory) / "rank_allocator_state.json").open("w", encoding="utf-8") as handle:
            json.dump(self.state_dict(), handle, indent=2)

    def load_state(self, directory):
        path = Path(directory) / "rank_allocator_state.json"
        if not path.is_file():
            return
        with path.open(encoding="utf-8") as handle:
            state = json.load(handle)
        self.event_count = int(state.get("event_count", 0))
        self.optimizer.id_ema = {}
        for name, value in state.get("id_ema", {}).items():
            layer_text, branch = name.split(".")
            self.optimizer.id_ema[BranchKey(int(layer_text.split("_")[1]), branch)] = float(value)
        if self.trainer.gradient_callback is not None:
            self.trainer.gradient_callback.gradient_ema = {}
            for name, value in state.get("gradient_ema", {}).items():
                layer_text, branch = name.split(".")
                key = BranchKey(int(layer_text.split("_")[1]), branch)
                self.trainer.gradient_callback.gradient_ema[key] = float(value)

    def _append_capacity(self, step, geometry):
        path = self.metrics_dir / "branch_capacity_all.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        gradients = (
            self.trainer.gradient_callback.gradient_ema
            if self.trainer.gradient_callback is not None
            else {}
        )
        with path.open("a", newline="", encoding="utf-8") as handle:
            fields = [
                "step", "layer", "branch", "active_rank", "effective_rank",
                "parameter_rank_saturation", "gradient_ema",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            if new_file:
                writer.writeheader()
            for row in geometry:
                writer.writerow(
                    {
                        "step": step,
                        "layer": row.layer,
                        "branch": row.branch,
                        "active_rank": row.active_rank,
                        "effective_rank": row.effective_rank,
                        "parameter_rank_saturation": row.parameter_rank_saturation,
                        "gradient_ema": gradients.get(row.key, 0.0),
                    }
                )

    def _update_id_ema(self, geometry):
        for row in geometry:
            previous = self.optimizer.id_ema.get(row.key, row.id_input_lcb)
            value = self.optimizer.ema_alpha * row.id_input_lcb + (1.0 - self.optimizer.ema_alpha) * previous
            row.id_input_ema = float(value)
            self.optimizer.id_ema[row.key] = float(value)

    def _append_global_metrics(self, step, geometry):
        path = self.metrics_dir / "global_metrics.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        valid_output_ids = [
            row.id_output_median
            for row in geometry
            if row.output_id_valid and torch.isfinite(torch.tensor(row.id_output_median))
        ]
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(["step", "mean_input_id_lcb", "mean_input_id_ema", "mean_output_id"])
            writer.writerow(
                [
                    step,
                    sum(row.id_input_lcb for row in geometry) / max(1, len(geometry)),
                    sum(row.id_input_ema for row in geometry) / max(1, len(geometry)),
                    sum(valid_output_ids) / len(valid_output_ids) if valid_output_ids else "",
                ]
            )

    def _append_runtime(self, step, started, accepted_transfers):
        path = self.metrics_dir / "runtime_overhead.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            if new_file:
                writer.writerow(["step", "event", "wall_time", "accepted_transfers"])
            writer.writerow([step, self.event_count, time.perf_counter() - started, accepted_transfers])

    def _reset_new_optimizer_moments(self, optimizer, before, after, model):
        controls = dict(iter_branch_controls(model))
        for key, new_rank in after.items():
            old_rank = before[key]
            if new_rank <= old_rank:
                continue
            control = controls[key]
            for parameter, axis in ((control.lora_A, 0), (control.lora_B, 1)):
                state = optimizer.state.get(parameter, {})
                for state_name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    tensor = state.get(state_name)
                    if tensor is None:
                        continue
                    slices = [slice(None)] * tensor.ndim
                    slices[axis] = slice(old_rank, new_rank)
                    tensor[tuple(slices)].zero_()

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        for _, branch in iter_branch_controls(model):
            branch.advance_transition()
        freeze_step = int(state.max_steps * (1.0 - self.freeze_ratio))
        if state.global_step >= freeze_step:
            if not self.trainer.rank_frozen:
                for _, branch in iter_branch_controls(model):
                    branch.set_active_rank(branch.active_rank, step=state.global_step)
            self.trainer.rank_frozen = True
            return
        if state.global_step == 0 or state.global_step % self.allocation_interval:
            return

        self.event_count += 1
        started = time.perf_counter()
        dataloader = self.trainer.get_rank_calibration_dataloader()
        geometry = measure_branch_geometry(
            model,
            dataloader,
            self.trainer.processing_class,
            step=state.global_step,
            sample_size=self.id_sample_size,
            bootstrap_repeats=self.bootstrap_repeats,
            bootstrap_fraction=self.bootstrap_fraction,
            uncertainty_lambda=self.uncertainty_lambda,
        )
        self._update_id_ema(geometry)
        append_geometry_csv(self.metrics_dir / "branch_geometry_all.csv", geometry)
        self._append_capacity(state.global_step, geometry)
        self._append_global_metrics(state.global_step, geometry)
        if self.event_count <= self.warmup_events:
            print(f"[ID-DR] geometry warmup event {self.event_count}/{self.warmup_events}")
            self._append_runtime(state.global_step, started, 0)
            return

        before = get_rank_map(model)
        after, prior, transfers = self.optimizer.run_event(
            model,
            dataloader,
            before,
            geometry,
            state.global_step,
            self.probe_examples,
            self.direct_examples,
            self.metrics_dir,
            gradient_ema=(
                self.trainer.gradient_callback.gradient_ema
                if self.trainer.gradient_callback is not None
                else None
            ),
        )
        set_rank_map(model, after, step=state.global_step, transition_steps=self.transition_steps)
        self._reset_new_optimizer_moments(optimizer, before, after, model)
        self._append_runtime(
            state.global_step, started, sum(transfer.accepted for transfer in transfers)
        )
