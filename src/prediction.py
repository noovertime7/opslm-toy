import torch


def get_next_token_predictions(
    logits: torch.Tensor,
) -> torch.Tensor:
    """
    根据模型 logits 获取每个位置预测概率最高的 Token ID。

    Args:
        logits:
            [B, T, V]

    Returns:
        predicted_ids:
            [B, T]
    """

    if logits.ndim != 3:
        raise ValueError(
            "logits must have shape [B, T, V]"
        )

    predicted_ids = torch.argmax(
        logits,
        dim=-1,
    )

    return predicted_ids


def get_topk_predictions(
    logits: torch.Tensor,
    k: int = 5,
):
    """
    获取每个位置 Top-K Token。

    Args:
        logits:
            [B, T, V]

        k:
            Top-K 数量

    Returns:
        values:
            [B, T, K]

        indices:
            [B, T, K]
    """

    if logits.ndim != 3:
        raise ValueError(
            "logits must have shape [B, T, V]"
        )

    if k <= 0:
        raise ValueError(
            "k must be greater than 0"
        )

    if k > logits.size(-1):
        raise ValueError(
            "k cannot be larger than vocab size"
        )

    values, indices = torch.topk(
        logits,
        k=k,
        dim=-1,
    )

    return values, indices
