# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Iterator, Optional

import torch
from torch.utils.data import RandomSampler, Sampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn
from .config import DataConfig


class OrderedSampler(Sampler[int]):
    """
    Sample the rows listed in a jsonl file of `row_id`s in the given order
    """

    def __init__(self, dataset: RLHFDataset, order_file: str, batch_size: int):
        with open(order_file, encoding="utf-8") as f:
            row_ids = [json.loads(line)["row_id"] for line in f if line.strip()]

        if len(row_ids) != len(set(row_ids)):
            raise ValueError(f"Duplicate row ids found in {order_file}.")

        row_id2index = {row_id: index for index, row_id in enumerate(dataset.row_ids)}
        missing_row_ids = [row_id for row_id in row_ids if row_id not in row_id2index]
        if len(missing_row_ids) != 0:
            raise ValueError(f"{len(missing_row_ids)} row ids in {order_file} are not in the dataset.")

        indices = [row_id2index[row_id] for row_id in row_ids]
        self.indices = indices[: len(indices) // batch_size * batch_size]  # drop last

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> None:
    train_dataset = RLHFDataset(
        data_path=config.train_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        system_prompt=config.system_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
    )
    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    # use sampler for better ckpt resume
    if config.order_file is not None:
        sampler = OrderedSampler(train_dataset, config.order_file, batch_size=train_batch_size)
    elif config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=config.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    val_dataset = RLHFDataset(
        data_path=config.val_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        system_prompt=config.system_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
    )

    if config.val_batch_size == -1:
        val_batch_size = len(val_dataset)
    else:
        val_batch_size = config.val_batch_size

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    assert len(train_dataloader) >= 1
    assert len(val_dataloader) >= 1
    print(f"Size of train dataloader: {len(train_dataloader)}")
    print(f"Size of val dataloader: {len(val_dataloader)}")
    return train_dataloader, val_dataloader
