import os
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

from aesl_loss import AESLLoss
from sft_dataset import IGNORE_INDEX, SFTCollator, SFTDataset


@dataclass
class ColdStartConfig:
    enable: bool = True
    model_path: Optional[str] = None
    output_dir: str = "./checkpoints/cold_start"
    experiment_name: str = "cold_start"
    logger: List[str] = field(default_factory=lambda: ["none"])

    # Data
    train_files: str = ""
    thinking_file: Optional[str] = None
    prompt_key: str = "problem"
    answer_key: str = "solution"
    image_key: str = "image"
    format_prompt: Optional[str] = None
    system_prompt: Optional[str] = None
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    max_seq_length: int = 4864
    dataloader_num_workers: int = 8

    # Loss: ce, wce (answer-weighted), aesl, wce_aesl
    loss_type: str = "ce"
    answer_weight: float = 0.6
    aesl_temperature: float = 5.0

    # Optimization
    lr: float = 1.0e-5
    lr_scheduler_type: str = "cosine"
    lr_warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_train_epochs: float = 1.0
    max_steps: int = -1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    seed: int = 42
    attn_implementation: str = "flash_attention_2"
    fsdp: str = ""

    # Checkpoints
    save_steps: int = 200
    save_limit: int = 1


class ColdStartTrainer(Trainer):
    """Trainer with answer-weighted cross entropy and AESL on the target tokens."""

    def __init__(self, *args, loss_type="ce", answer_weight=0.6, aesl_temperature=5.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.answer_weight = answer_weight
        self.aesl_loss = AESLLoss(temperature_scaling=aesl_temperature)
        self.num_kept_tokens = 0
        self.num_target_tokens = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        answer_mask = inputs.pop("answer_mask")
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)

        if model.training and self.loss_type != "ce":
            # Align logits, labels and the answer mask to the predicted tokens
            logits = outputs.logits[:, :-1]
            labels = inputs["labels"][:, 1:]
            answer_mask = answer_mask[:, 1:].bool()
            if self.loss_type == "wce":
                loss = self.weighted_ce_loss(logits, labels, answer_mask, num_items_in_batch)
            else:
                loss = self.aesl_ce_loss(logits, labels, answer_mask)

        return (loss, outputs) if return_outputs else loss

    def weighted_ce_loss(self, logits, labels, answer_mask, num_items_in_batch=None):
        """Cross entropy with separate weights on the answer tokens and the remaining target tokens."""
        loss_mask = labels != IGNORE_INDEX
        token_loss = F.cross_entropy(logits[loss_mask].float(), labels[loss_mask], reduction="none")
        is_answer = answer_mask[loss_mask]

        num_tokens = token_loss.numel()
        num_answer_tokens = int(is_answer.sum())
        num_other_tokens = num_tokens - num_answer_tokens
        weights = torch.ones_like(token_loss)
        if num_answer_tokens > 0 and num_other_tokens > 0:
            # Weights sum to the number of target tokens, so the loss keeps the scale of cross entropy
            weights[is_answer] = self.answer_weight * num_tokens / num_answer_tokens
            weights[~is_answer] = (1.0 - self.answer_weight) * num_tokens / num_other_tokens

        if num_items_in_batch is None:
            num_items_in_batch = max(num_tokens, 1)

        return (weights * token_loss).sum() / float(num_items_in_batch)

    def aesl_ce_loss(self, logits, labels, answer_mask):
        """AESL on the target tokens, wce_aesl averages the kept answer tokens and the other kept tokens separately."""
        zero = logits.sum() * 0.0
        loss_sum, answer_loss_sum, other_loss_sum = zero, zero, zero
        num_kept, num_kept_answer, num_kept_other = 0, 0, 0
        for i in range(labels.shape[0]):
            loss_mask = labels[i] != IGNORE_INDEX
            if not loss_mask.any():
                continue

            # Only the target tokens are passed to AESL, every one of them is supervised
            token_logits = logits[i][loss_mask].float()
            token_labels = labels[i][loss_mask]
            token_loss = F.cross_entropy(token_logits, token_labels, reduction="none")
            compute_loss_mask = self.aesl_loss.compute_loss_mask(
                -token_loss[None], token_logits[None], token_labels[None], torch.ones_like(token_loss)[None]
            )[0]

            kept_answer = compute_loss_mask & answer_mask[i][loss_mask]
            kept_other = compute_loss_mask & ~answer_mask[i][loss_mask]
            loss_sum = loss_sum + (token_loss * compute_loss_mask).sum()
            answer_loss_sum = answer_loss_sum + (token_loss * kept_answer).sum()
            other_loss_sum = other_loss_sum + (token_loss * kept_other).sum()
            num_kept += int(compute_loss_mask.sum())
            num_kept_answer += int(kept_answer.sum())
            num_kept_other += int(kept_other.sum())
            self.num_target_tokens += int(loss_mask.sum())

        self.num_kept_tokens += num_kept
        if self.loss_type == "wce_aesl" and num_kept_answer > 0 and num_kept_other > 0:
            answer_loss = self.answer_weight * answer_loss_sum / num_kept_answer
            other_loss = (1.0 - self.answer_weight) * other_loss_sum / num_kept_other
            return answer_loss + other_loss

        return loss_sum / max(num_kept, 1)

    def log(self, logs, *args, **kwargs):
        # Report the fraction of target tokens kept by AESL since the last log
        if self.num_target_tokens > 0:
            logs["aesl_keep_frac"] = self.num_kept_tokens / self.num_target_tokens
            self.num_kept_tokens, self.num_target_tokens = 0, 0

        super().log(logs, *args, **kwargs)


def main():
    cli_args = OmegaConf.from_cli()
    file_config = OmegaConf.load(cli_args.pop("config"))
    file_config = OmegaConf.merge(file_config, cli_args)
    OmegaConf.resolve(file_config)
    config = OmegaConf.merge(OmegaConf.structured(ColdStartConfig()), file_config.cold_start)
    config: ColdStartConfig = OmegaConf.to_object(config)
    if not config.enable:
        print("Cold start is disabled in the config, nothing to do.")
        return

    if config.loss_type not in ("ce", "wce", "aesl", "wce_aesl"):
        raise ValueError(f"Unknown loss type: {config.loss_type}.")

    processor = AutoProcessor.from_pretrained(config.model_path)
    train_dataset = SFTDataset(
        data_path=config.train_files,
        thinking_file=config.thinking_file,
        processor=processor,
        format_prompt=config.format_prompt,
        system_prompt=config.system_prompt,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        max_seq_length=config.max_seq_length,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        config.model_path, torch_dtype=torch.bfloat16, attn_implementation=config.attn_implementation
    )
    model.config.use_cache = False

    fsdp_kwargs = {}
    if config.fsdp:
        fsdp_kwargs["fsdp"] = config.fsdp
        fsdp_kwargs["fsdp_config"] = {
            "transformer_layer_cls_to_wrap": list(model._no_split_modules),
            "cpu_ram_efficient_loading": True,
            "sync_module_states": True,
        }

    training_args = TrainingArguments(
        output_dir=config.output_dir,
        run_name=config.experiment_name,
        report_to=config.logger,
        num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.lr,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.lr_warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_limit,
        save_only_model=True,
        seed=config.seed,
        data_seed=config.seed,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        **fsdp_kwargs,
    )

    trainer = ColdStartTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=SFTCollator(pad_token_id=processor.tokenizer.pad_token_id),
        loss_type=config.loss_type,
        answer_weight=config.answer_weight,
        aesl_temperature=config.aesl_temperature,
    )
    trainer.train()

    # The final model is the starting point of RL
    save_dir = os.path.join(config.output_dir, "final")
    trainer.save_model(save_dir)
    if trainer.is_world_process_zero():
        processor.save_pretrained(save_dir)


if __name__ == "__main__":
    main()
