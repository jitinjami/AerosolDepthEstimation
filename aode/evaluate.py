import torch


def evaluate(preds: list[float], gts: list[float]) -> dict:
    # TODO: Evaluate results. Output is a dictionary in the format of {'metric_name': value}.
    assert len(preds) == len(gts), (
        f'The length of predictions {len(preds)} does not match the length of'
        'ground truths.'
    )
    num_val = len(preds)
    preds = torch.tensor(preds)
    gts = torch.tensor(gts)

    up = num_val * (preds * gts).sum() - preds.sum() * gts.sum()
    down = torch.sqrt(
        num_val * (gts**2).sum() - (gts.sum()) ** 2
    ) * torch.sqrt(num_val * (preds**2).sum() - (preds.sum()) ** 2)

    return up / down
