# lora.py
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Implements a Low-Rank Adapter (LoRA) for Linear layers.
    W' = W + (alpha / r) * (B @ A)
    """
    def __init__(self, in_features, out_features, r=4, alpha=1.0, bias=False):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        if r > 0:
            self.A = nn.Parameter(torch.randn(r, in_features) * 0.01)
            self.B = nn.Parameter(torch.randn(out_features, r) * 0.01)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        if self.r > 0:
            lora_out = (x @ self.A.t()) @ self.B.t()
            lora_out = lora_out * self.scaling
            if self.bias is not None:
                lora_out = lora_out + self.bias
            return lora_out
        else:
            return torch.zeros_like(x)


class LoRAWrapper(nn.Module):
    """
    Wraps an existing Linear layer with LoRA adapter.
    Keeps original weights frozen and adds low-rank residual.
    """
    def __init__(self, base_layer: nn.Linear, r=8, alpha=16.0):
        super().__init__()
        self.base = base_layer
        self.lora = LoRALinear(base_layer.in_features, base_layer.out_features, r=r, alpha=alpha)

        # Freeze original linear weights
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.base(x) + self.lora(x)


def inject_lora_to_last_block(model: nn.Module, r=8, alpha=16.0):
    """
    Automatically finds the last transformer block in ViT model
    and injects LoRA adapters into its attention and MLP Linear layers.
    """
    if hasattr(model, "blocks"):
        last_block = model.blocks[-1]
    elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        last_block = model.encoder.layers[-1]
    else:
        raise RuntimeError("Could not locate transformer blocks for LoRA injection.")

    wrapped_layers = []
    for name, module in last_block.named_children():
        for sub_name, sub_module in module.named_children():
            if isinstance(sub_module, nn.Linear):
                wrapped_layers.append(f"{name}.{sub_name}")
                setattr(module, sub_name, LoRAWrapper(sub_module, r=r, alpha=alpha))

    # Fallback if no linear found directly
    if not wrapped_layers:
        for name, sub_module in last_block.named_modules():
            if isinstance(sub_module, nn.Linear):
                setattr(last_block, name, LoRAWrapper(sub_module, r=r, alpha=alpha))
                wrapped_layers.append(name)

    print(f"[LoRA] Injected adapters into last block layers: {wrapped_layers}")
    return wrapped_layers
