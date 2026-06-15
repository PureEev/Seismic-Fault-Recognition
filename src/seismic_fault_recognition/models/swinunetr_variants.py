"""Final checkpoint-compatible Swin Tiny architecture."""

from __future__ import annotations

from types import MethodType
from typing import Any, Sequence

from seismic_fault_recognition.registry import MODEL_REGISTRY


SWIN_TINY_INPUT_SIZE = (128, 128, 128)
SWIN_TINY_FEATURE_SIZE = 48


@MODEL_REGISTRY.register("swin_tiny")
def build_swinunetr_tiny(
    in_channels: int = 1,
    out_channels: int = 1,
    img_size: Sequence[int] = SWIN_TINY_INPUT_SIZE,
    use_checkpoint: bool = True,
    **_: Any,
) -> Any:
    """Build the single Swin Tiny model used by notebooks and tests."""

    input_size = tuple(int(value) for value in img_size)
    if input_size != SWIN_TINY_INPUT_SIZE:
        raise ValueError(
            f"Swin Tiny requires img_size={SWIN_TINY_INPUT_SIZE}, got {input_size}"
        )

    try:
        from monai.networks.nets import SwinUNETR
    except ImportError as exc:  # pragma: no cover - training environments include MONAI.
        raise ImportError("MONAI is required for Swin Tiny") from exc

    model = SwinUNETR(
        in_channels=int(in_channels),
        out_channels=int(out_channels),
        patch_size=2,
        depths=(2, 2, 2, 1),
        num_heads=(3, 6, 12, 24),
        window_size=(7, 7, 7),
        qkv_bias=True,
        mlp_ratio=4.0,
        feature_size=SWIN_TINY_FEATURE_SIZE,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.1,
        patch_norm=True,
        use_checkpoint=bool(use_checkpoint),
        spatial_dims=3,
    )
    return finalize_swinunetr_tiny(model)


def finalize_swinunetr_tiny(model: Any) -> Any:
    """Apply the patch-5/no-deep architecture used by the final checkpoint."""

    _, nn, F = _require_torch_modules()
    if not hasattr(model, "swinViT") or not hasattr(model.swinViT, "patch_embed"):
        raise AttributeError("model must expose model.swinViT.patch_embed")

    patch_embed = model.swinViT.patch_embed
    projection = getattr(patch_embed, "proj", None)
    in_channels = int(getattr(projection, "in_channels", 1))
    out_channels = int(
        getattr(projection, "out_channels", getattr(patch_embed, "embed_dim", SWIN_TINY_FEATURE_SIZE))
    )
    if out_channels != SWIN_TINY_FEATURE_SIZE:
        raise ValueError(
            f"Swin Tiny requires feature_size={SWIN_TINY_FEATURE_SIZE}, got {out_channels}"
        )

    patch_embed.proj = nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=5,
        stride=2,
        padding=2,
        bias=False,
    )

    if hasattr(model.swinViT, "layers4"):
        model.swinViT.layers4 = nn.Identity()
    if hasattr(model.swinViT, "num_layers"):
        model.swinViT.num_layers = 3
    for name in ("encoder10", "decoder5"):
        if hasattr(model, name):
            delattr(model, name)

    model.enc2to3 = nn.Conv3d(48, 96, kernel_size=1, bias=False)
    model.enc3to4 = nn.Conv3d(96, 192, kernel_size=1, bias=False)
    nn.init.kaiming_normal_(model.enc2to3.weight, nonlinearity="relu")
    nn.init.kaiming_normal_(model.enc3to4.weight, nonlinearity="relu")

    def _forward(self: Any, x: Any) -> Any:
        spatial_size = tuple(int(value) for value in x.shape[-3:])
        if spatial_size != SWIN_TINY_INPUT_SIZE:
            raise ValueError(
                f"Swin Tiny requires input spatial size {SWIN_TINY_INPUT_SIZE}, got {spatial_size}"
            )

        enc1 = self.encoder1(x)
        enc2 = self.encoder2(F.max_pool3d(enc1, kernel_size=2, stride=2))
        enc3 = self.encoder3(
            self.enc2to3(F.max_pool3d(enc2, kernel_size=2, stride=2))
        )
        enc4 = self.encoder4(
            self.enc3to4(F.max_pool3d(enc3, kernel_size=2, stride=2))
        )

        dec3 = self.decoder3(enc4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)
        return self.out(dec1)

    model.forward = MethodType(_forward, model)
    return model


def _require_torch_modules() -> tuple[Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:  # pragma: no cover - training environments include PyTorch.
        raise ImportError("PyTorch is required for Swin Tiny") from exc
    return torch, nn, F
