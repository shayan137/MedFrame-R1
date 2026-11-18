import torch
import torch.nn as nn


class AESLLoss(nn.Module):
    """
    Adaptive Early-Stop Loss, adapted from the official code of
    "Getting Your LLMs Ready for Reinforcement Learning with Lightweight SFT" (openrlhf/models/loss.py)
    """

    def __init__(self, token_level_loss: bool = True, temperature_scaling=1):
        super().__init__()
        assert token_level_loss, "Non-token level loss not implemented"
        self.token_level_loss = token_level_loss
        self.temperature_scaling = temperature_scaling

    def compute_loss_mask(self, per_token_logps: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        """Select the tokens to compute loss, inputs are aligned to the predicted tokens."""
        with torch.no_grad():
            cumsum_neg_log_probs = torch.cumsum(-per_token_logps * loss_mask, dim=1)  # (bs, seq_len-1)
            cumsum_loss_mask = torch.cumsum(loss_mask, dim=1)  # (bs, seq_len-1)
            # Shift cumulative sums to exclude the current token
            # always not mask the first token
            shifted_cumsum_neg_log_probs = torch.cat([torch.full_like(cumsum_neg_log_probs[:, :1], float('inf')), cumsum_neg_log_probs[:, :-1]], dim=1)
            # Replace any 0 in shifted_cumsum_neg_log_probs with inf
            shifted_cumsum_neg_log_probs[shifted_cumsum_neg_log_probs == 0] = float('inf')
            prefix_mean_neg_log_probs = shifted_cumsum_neg_log_probs / (cumsum_loss_mask + 1e-8)  # (bs, seq_len-1)
            prefix_mean_neg_log_probs = prefix_mean_neg_log_probs.clamp(min=0, max=10)
            logits = logits / (self.temperature_scaling * prefix_mean_neg_log_probs[:,:, None]).clamp(min=1e-8)

            # Compute only exp(logits[label]) and the normalization denominator
            max_logits = torch.amax(logits, dim=-1, keepdim=True)  # (bs, seq_len-1, 1)
            label_logits = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)  # (bs, seq_len-1)
            label_exp_logits = torch.exp(label_logits - max_logits.squeeze(-1))  # (bs, seq_len-1)
            # Denominator: sum(exp(logits)) computed in chunks
            vocab_chunk_size = 10000  # Process logits in smaller chunks for memory efficiency
            denom = torch.zeros_like(label_logits)  # Initialize denominator
            for chunk_start in range(0, logits.size(-1), vocab_chunk_size):
                chunk_logits = logits[:, :, chunk_start:chunk_start + vocab_chunk_size]  # Get a chunk of vocab
                denom += torch.sum(torch.exp(chunk_logits - max_logits), dim=-1)
            # Compute label probabilities
            label_probs = label_exp_logits / denom  # (bs, seq_len-1)

            condition_mask = torch.rand_like(label_probs) < label_probs
            compute_loss_mask = (~condition_mask) & (loss_mask.bool())  # (bs, seq_len-1), bool
        return compute_loss_mask

    def forward(self, per_token_logps: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:

        logits = logits[:, :-1]
        labels = labels[:, :-1]

        compute_loss_mask = self.compute_loss_mask(per_token_logps, logits, labels, loss_mask)
        loss = (-per_token_logps * compute_loss_mask).sum(dim=None) / (compute_loss_mask.sum(dim=None) + 1e-8)

        return loss
