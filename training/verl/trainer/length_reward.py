"""
Length reward
"""

from collections import defaultdict

import torch

from ..protocol import DataProto


def apply_length_reward(
    data: DataProto,
    threshold: float,
    decay: float,
    max_length: float,
    neg_scale: float,
    correct_threshold: float,
) -> dict[str, float]:
    """Rescale the group-normalized advantages by response length, the sign of advantages is unchanged.

    Correct responses with positive advantages are weighted by exp(-(length - threshold) / decay), and the weights
    are normalized by their mean in each group, so that shorter correct responses get larger advantages.
    Incorrect responses with negative advantages are scaled by 1 + neg_scale * (length - threshold) / (max_length -
    threshold) once they exceed the threshold.

    Args:
        data: `(DataProto)`
            requires `advantages`, `response_mask` and `correctness` of shape (bs, response_length), (bs,
            response_length) and (bs,), and `uid` in the non tensor batch
        threshold: `(float)`
            response length in tokens below which the advantages are unchanged
        decay: `(float)`
            decay length in tokens of the weight above the threshold
        max_length: `(float)`
            max response length in tokens
        neg_scale: `(float)`
            max extra scale on the negative advantages of incorrect responses
        correct_threshold: `(float)`
            a response is correct if its correctness reaches this value

    Returns:
        metrics: `(dict[str, float])`
    """
    advantages = data.batch["advantages"]
    response_mask = data.batch["response_mask"].to(advantages.dtype)
    response_length = response_mask.sum(dim=-1)  # (bs,)
    # the advantage is the same on every response token
    scores = (advantages * response_mask).sum(dim=-1) / response_length.clamp(min=1.0)
    is_correct = data.batch["correctness"].to(advantages.device) >= correct_threshold

    weights = torch.where(
        response_length <= threshold,
        torch.ones_like(response_length),
        torch.exp(-(response_length - threshold) / decay),
    )
    factors = torch.ones_like(response_length)

    is_positive = is_correct & (scores > 0)
    uid2idxs = defaultdict(list)
    for idx, uid in enumerate(data.non_tensor_batch["uid"]):
        uid2idxs[uid].append(idx)

    for idxs in uid2idxs.values():
        idxs = torch.tensor(idxs, dtype=torch.long, device=advantages.device)
        idxs = idxs[is_positive[idxs]]
        if len(idxs) != 0:
            factors[idxs] = weights[idxs] / weights[idxs].mean()

    is_negative = (~is_correct) & (scores < 0) & (response_length > threshold)
    overlong_ratio = (response_length - threshold) / max(max_length - threshold, 1.0)
    factors[is_negative] = 1.0 + neg_scale * overlong_ratio[is_negative].clamp(max=1.0)

    data.batch["advantages"] = advantages * factors.unsqueeze(-1)

    def mean(tensor: torch.Tensor, default: float) -> float:
        return tensor.mean().item() if tensor.numel() != 0 else default

    return {
        "length_reward/pos_frac": is_positive.float().mean().item(),
        "length_reward/neg_frac": is_negative.float().mean().item(),
        "length_reward/pos_factor": mean(factors[is_positive], 1.0),
        "length_reward/neg_factor": mean(factors[is_negative], 1.0),
        "length_reward/overlong_frac": (response_length > threshold).float().mean().item(),
        "length_reward/correct_length": mean(response_length[is_correct], 0.0),
        "length_reward/incorrect_length": mean(response_length[~is_correct], 0.0),
    }
