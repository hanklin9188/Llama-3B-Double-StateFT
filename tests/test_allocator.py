import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader
from transformers import LlamaConfig, LlamaForCausalLM, TrainingArguments, default_data_collator

from pcft.metrics.adaptive_rank import IDRegularizedRankExchangeOptimizer
from pcft.metrics.branch_geometry import measure_branch_geometry
from pcft.rank import get_rank_map
from pcft.trainer import BranchGradientEMACallback, DynamicRankCallback, IDDRTrainer
from pcft.wrapping.wrap import wrap_model_with_double_control


class TokenizerStub:
    eos_token_id = 2


def collate(rows):
    return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}


class AllocatorIntegrationTests(unittest.TestCase):
    @staticmethod
    def model_and_rows():
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        config.use_cache = False
        model = wrap_model_with_double_control(
            LlamaForCausalLM(config),
            ranks_attn=[16, 16],
            ranks_mlp=[16, 16],
            rank_max_attn=[24, 24],
            rank_max_mlp=[24, 24],
            alpha=1.0,
            dropout=0.0,
            rank_quantum=8,
            checkpoint_base=False,
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for layer in model.model.layers:
            for branch in (layer.ctrl_attn, layer.ctrl_mlp):
                branch.lora_A.requires_grad_(True)
                branch.lora_B.requires_grad_(True)
        rows = []
        generator = torch.Generator().manual_seed(42)
        for _ in range(16):
            input_ids = torch.randint(3, 64, (12,), generator=generator)
            rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": torch.ones(12, dtype=torch.long),
                    "labels": input_ids.clone(),
                }
            )
        return model, rows

    def test_geometry_probe_and_transfer_pipeline(self):
        model, rows = self.model_and_rows()
        dataloader = DataLoader(rows, batch_size=8, collate_fn=collate)
        stable_id = {"median": 8.0, "mad": 0.5, "lcb": 7.5}
        with patch(
            "pcft.metrics.branch_geometry.estimate_id_with_bootstrap",
            return_value=stable_id,
        ):
            geometry = measure_branch_geometry(
                model, dataloader, TokenizerStub(), step=1, sample_size=16
            )
        self.assertEqual(len(geometry), 4)

        allocator = IDRegularizedRankExchangeOptimizer(
            rank_min=8,
            rank_max=24,
            rank_quantum=8,
            global_budget=64,
            switch_cost=0.0,
            move_threshold=-1.0,
            receiver_count=4,
            donor_count=4,
            direct_verify_pairs=2,
            max_transfers=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            after, _, transfers = allocator.run_event(
                model,
                dataloader,
                get_rank_map(model),
                geometry,
                step=1,
                probe_examples=8,
                direct_examples=8,
                metrics_dir=directory,
            )
        self.assertEqual(sum(after.values()), 64)
        self.assertEqual(len(transfers), 1)
        self.assertTrue(transfers[0].accepted)

    def test_trainer_runs_dynamic_callback(self):
        model, rows = self.model_and_rows()
        control_config = {
            "architecture": "id_dr_double_control_v2",
            "hidden_size": 32,
            "rank_min": 8,
            "rank_max": 24,
            "rank_quantum": 8,
            "global_rank_budget": 64,
            "alpha": 1.0,
            "scaling_mode": "alpha_over_sqrt_rank",
        }
        with tempfile.TemporaryDirectory() as directory:
            arguments = TrainingArguments(
                output_dir=directory,
                per_device_train_batch_size=4,
                per_device_eval_batch_size=8,
                max_steps=2,
                learning_rate=1e-3,
                save_strategy="no",
                eval_strategy="no",
                logging_strategy="no",
                report_to="none",
                remove_unused_columns=False,
                disable_tqdm=True,
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            trainer = IDDRTrainer(
                model=model,
                args=arguments,
                train_dataset=rows,
                eval_dataset=rows,
                rank_calibration_dataset=rows,
                data_collator=default_data_collator,
                processing_class=None,
                control_config=control_config,
                exploration_probability=0.0,
            )
            gradient = BranchGradientEMACallback()
            trainer.gradient_callback = gradient
            trainer.add_callback(gradient)
            dynamic = DynamicRankCallback(
                trainer,
                allocation_interval=1,
                warmup_events=0,
                freeze_ratio=0.0,
                id_sample_size=16,
                bootstrap_repeats=1,
                probe_examples=8,
                direct_examples=8,
                rank_min=8,
                rank_max=24,
                rank_quantum=8,
                global_budget=64,
                switch_cost=0.0,
                move_threshold=-1.0,
                receiver_count=4,
                donor_count=4,
                direct_verify_pairs=2,
                max_transfers=1,
            )
            trainer.dynamic_callback = dynamic
            trainer.add_callback(dynamic)
            stable_id = {"median": 8.0, "mad": 0.5, "lcb": 7.5}
            with patch(
                "pcft.metrics.branch_geometry.estimate_id_with_bootstrap",
                return_value=stable_id,
            ):
                trainer.train()
            self.assertTrue((torch.tensor(list(get_rank_map(model).values())) >= 8).all())
            self.assertTrue((Path(directory) / "metrics" / "rank_all.csv").is_file())


if __name__ == "__main__":
    unittest.main()
