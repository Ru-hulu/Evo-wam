import torch


def create_custom_forward(module):
    def custom_forward(*inputs, **kwargs):
        return module(*inputs, **kwargs)

    return custom_forward


def gradient_checkpoint_forward(model, use_gradient_checkpointing, *args, **kwargs):
    if use_gradient_checkpointing:
        return torch.utils.checkpoint.checkpoint(
            create_custom_forward(model),
            *args,
            **kwargs,
            use_reentrant=False,
        )
    return model(*args, **kwargs)
