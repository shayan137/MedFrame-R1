# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Based on:
# https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
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

from typing import Optional

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration, Qwen3_5Model


def _get_visual_embeds(model: "Qwen3_5Model", pixel_values: torch.Tensor, grid_thw: torch.LongTensor):
    outputs = model.visual(pixel_values, grid_thw=grid_thw, return_dict=True)
    visual_embeds = outputs.pooler_output
    if isinstance(visual_embeds, (list, tuple)):
        visual_embeds = torch.cat(visual_embeds, dim=0)

    return visual_embeds


def _get_input_embeds(
    model: "Qwen3_5Model",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        image_embeds = _get_visual_embeds(model, pixel_values, image_grid_thw)
        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == model.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        video_embeds = _get_visual_embeds(model, pixel_values_videos, video_grid_thw)
        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        mask = input_ids == model.config.video_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        video_mask = mask_expanded.to(inputs_embeds.device)

        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}


def qwen3_5_base_forward(
    self: "Qwen3_5Model",
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    if input_ids is not None:
        input_kwargs = _get_input_embeds(
            self,
            input_ids,
            attention_mask,
            kwargs.pop("pixel_values", None),
            kwargs.pop("pixel_values_videos", None),
            kwargs.pop("image_grid_thw", None),
            kwargs.pop("video_grid_thw", None),
        )
        kwargs.update(input_kwargs)  # avoid lora module to have multiple keyword arguments
        input_ids = None

    return self.language_model(input_ids=input_ids, **kwargs)


def qwen3_5_model_forward(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: Optional[torch.LongTensor] = None,
    **kwargs,
) -> "CausalLMOutputWithPast":
    outputs = self.model(input_ids=input_ids, **kwargs)
    hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
    logits = self.lm_head(hidden_states)

    return CausalLMOutputWithPast(logits=logits)
