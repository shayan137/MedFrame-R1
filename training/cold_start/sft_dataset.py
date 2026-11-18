import glob
import json
import os
from typing import Any, Optional

import torch
from datasets import load_dataset, load_from_disk
from jinja2 import Template
from torch.utils.data import Dataset

from verl.utils.dataset import process_image


IGNORE_INDEX = -100


def build_messages(prompt_str: str, has_image: bool, system_prompt: Optional[str] = None) -> list[dict[str, Any]]:
    """Build the chat messages in the same way as the RL dataset."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

    if not has_image:
        return messages + [{"role": "user", "content": prompt_str}]

    content_list = []
    for i, content in enumerate(prompt_str.split("<image>")):
        if i != 0:
            content_list.append({"type": "image"})

        if content:
            content_list.append({"type": "text", "text": content})

    return messages + [{"role": "user", "content": content_list}]


def make_target(solution: str, thinking: str) -> str:
    """Build the target response from the modality tag, the reasoning and the answer."""
    modality, answer = "", solution.strip()
    if ">" in solution:
        modality, answer = solution.split(">", maxsplit=1)
        modality, answer = modality + ">", answer.strip()

    return f"{modality}\n<thinking>\n{thinking.strip()}\n</thinking>\n<answer>{answer}</answer>"


class SFTDataset(Dataset):
    """Dataset for the supervised cold start, the loss is computed on the target response only."""

    def __init__(
        self,
        data_path: str,
        thinking_file: str,
        processor,
        format_prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
        prompt_key: str = "problem",
        answer_key: str = "solution",
        image_key: str = "image",
        min_pixels: Optional[int] = 262144,
        max_pixels: Optional[int] = 4194304,
        max_seq_length: int = 4864,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_seq_length = max_seq_length

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            data_files = sorted(glob.glob(os.path.join(data_path, "data", f"{data_split}-*.parquet")))
            if data_files:
                self.dataset = load_dataset("parquet", data_files=data_files, split="train")
            else:
                self.dataset = load_from_disk(data_path)[data_split]
        else:
            self.dataset = load_dataset(data_path, split=data_split)

        # Row ids follow the RL dataset, they link each sample to its reasoning
        if "row_id" not in self.dataset.column_names:
            self.dataset = self.dataset.add_column("row_id", list(range(len(self.dataset))))

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = Template(f.read().strip())

        self.system_prompt = None
        if system_prompt:
            with open(system_prompt, encoding="utf-8") as f:
                self.system_prompt = f.read().strip()

        self.thinking = {}
        with open(thinking_file, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    self.thinking[record["row_id"]] = record["thinking"]

        # Keep the samples that have a reasoning target
        kept_indices = [i for i, row_id in enumerate(self.dataset["row_id"]) if row_id in self.thinking]
        self.dataset = self.dataset.select(kept_indices)
        print(f"Cold Start Dataset Size: {len(self.dataset)}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.dataset[index]
        prompt_str = example[self.prompt_key]
        if self.format_prompt:
            prompt_str = self.format_prompt.render(content=prompt_str)

        messages = build_messages(prompt_str, self.image_key in example, self.system_prompt)
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        images = example.get(self.image_key) or []
        images = [process_image(image, self.min_pixels, self.max_pixels) for image in images]
        model_inputs = self.processor(images=images or None, text=[prompt], add_special_tokens=False, return_tensors="pt")
        prompt_ids = model_inputs.pop("input_ids")[0]
        model_inputs.pop("attention_mask")

        # Tokenize the reasoning part and the answer part separately to get the exact answer span
        target = make_target(example[self.answer_key], self.thinking[example["row_id"]])
        answer_start = target.rfind("<answer>") + len("<answer>")
        reasoning_ids = self.tokenizer(target[:answer_start], add_special_tokens=False)["input_ids"]
        answer_ids = self.tokenizer(target[answer_start:], add_special_tokens=False)["input_ids"]
        target_ids = reasoning_ids + answer_ids + [self.tokenizer.eos_token_id]
        answer_mask = [0] * len(reasoning_ids) + [1] * (len(answer_ids) + 1)

        # Truncate the target only, the prompt holds the image tokens
        max_target_length = max(self.max_seq_length - len(prompt_ids), 1)
        if len(target_ids) > max_target_length:
            target_ids = target_ids[: max_target_length - 1] + [self.tokenizer.eos_token_id]
            answer_mask = answer_mask[: max_target_length - 1] + [1]

        target_ids = torch.tensor(target_ids, dtype=prompt_ids.dtype)
        answer_mask = torch.tensor(answer_mask, dtype=prompt_ids.dtype)
        input_ids = torch.cat([prompt_ids, target_ids])
        features = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": torch.cat([torch.full_like(prompt_ids, IGNORE_INDEX), target_ids]),
            "answer_mask": torch.cat([torch.zeros_like(prompt_ids), answer_mask]),
        }

        # Extend the token type ids of the processor to the target tokens
        mm_token_type_ids = model_inputs.pop("mm_token_type_ids", None)
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids.reshape(-1)
            target_type_ids = torch.full((len(target_ids),), mm_token_type_ids[-1].item(), dtype=mm_token_type_ids.dtype)
            features["mm_token_type_ids"] = torch.cat([mm_token_type_ids, target_type_ids])

        features.update(model_inputs)
        return features


class SFTCollator:
    """Right-pad the text tensors and concatenate the vision tensors."""

    def __init__(self, pad_token_id: int):
        self.pad_values = {"input_ids": pad_token_id, "attention_mask": 0, "labels": IGNORE_INDEX, "answer_mask": 0}

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {}
        for key in dict.fromkeys(key for feature in features for key in feature):
            values = [feature[key] for feature in features if key in feature]
            if key in self.pad_values or key == "mm_token_type_ids":
                padded_values = []
                for value in values:
                    pad_value = self.pad_values[key] if key in self.pad_values else value[-1].item()
                    padding = torch.full((max_length - len(value),), pad_value, dtype=value.dtype)
                    padded_values.append(torch.cat([value, padding]))

                batch[key] = torch.stack(padded_values)
            elif torch.is_tensor(values[0]):
                batch[key] = torch.cat(values, dim=0)

        return batch
