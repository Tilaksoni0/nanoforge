import inspect
import torch


def configure_optimizers(
    model,
    weight_decay,
    learning_rate,
    device_type,
):
    """
    Optimizer construction is deliberately outside the training loop.

    The loop should only:
        optimizer.zero_grad()
        optimizer.step()
        update LR
    """

    param_dict = {
        n: p
        for n, p in model.named_parameters()
        if p.requires_grad
    }

    decay_param = [
        p for p in param_dict.values()
        if p.dim() >= 2
    ]

    nodecay_param = [
        p for p in param_dict.values()
        if p.dim() < 2
    ]

    optim_groups = [
        {
            "params": decay_param,
            "weight_decay": weight_decay,
        },
        {
            "params": nodecay_param,
            "weight_decay": 0.0,
        },
    ]

    fused_available = (
        "fused"
        in inspect.signature(torch.optim.AdamW).parameters
    )

    use_fused = (
        fused_available
        and device_type == "cuda"
    )

    return torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        fused=use_fused,
    )
