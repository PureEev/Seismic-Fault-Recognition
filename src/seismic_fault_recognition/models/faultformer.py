"""Attention FaultFormer variants consolidated from architecture notebooks."""

from __future__ import annotations

from typing import Any, Sequence

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for FaultFormer models")


from seismic_fault_recognition.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("faultformer_attention")
def build_faultformer_attention(**kwargs: Any) -> Any:
    """Build the attention-based FaultFormer model."""
    return AttentionFaultFormerNet()

if torch is not None:

    class ChannelAttention3D(nn.Module):
        """Channel attention block for 3D feature maps."""

        def __init__(self, channels: int, reduction: int = 8) -> None:
            super().__init__()
            hidden = max(1, channels // reduction)
            self.mlp = nn.Sequential(
                nn.Conv3d(channels, hidden, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden, channels, kernel_size=1),
            )

        def forward(self, x: Any) -> Any:
            """Apply channel attention to ``B,C,D,H,W`` features."""

            avg = self.mlp(F.adaptive_avg_pool3d(x, 1))
            maxv = self.mlp(F.adaptive_max_pool3d(x, 1))
            return x * torch.sigmoid(avg + maxv)


    class SpatialAttention3D(nn.Module):
        """Spatial attention block for 3D feature maps."""

        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)

        def forward(self, x: Any) -> Any:
            """Apply spatial attention to ``B,C,D,H,W`` features."""

            avg = x.mean(dim=1, keepdim=True)
            maxv, _ = x.max(dim=1, keepdim=True)
            return x * torch.sigmoid(self.conv(torch.cat([avg, maxv], dim=1)))


    class CBAM3D(nn.Module):
        """3D CBAM block combining channel and spatial attention."""

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.channel = ChannelAttention3D(channels)
            self.spatial = SpatialAttention3D()

        def forward(self, x: Any) -> Any:
            """Apply channel and spatial attention."""

            return self.spatial(self.channel(x))


    class ResBlock3D(nn.Module):
        """Residual 3D convolution block with instance normalization."""

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
                nn.GELU(),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm3d(out_channels, affine=True),
            )
            self.skip = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            )

        def forward(self, x: Any) -> Any:
            """Return residual block output."""

            return F.gelu(self.conv(x) + self.skip(x))


    class AttentionFaultFormerEncoder(nn.Module):
        """Encoder stack that interleaves residual blocks and 3D CBAM."""

        def __init__(self, in_channels: int = 1, channels: Sequence[int] = (24, 48, 96, 192)) -> None:
            super().__init__()
            blocks = []
            prev = in_channels
            for channel in channels:
                blocks.append(nn.Sequential(ResBlock3D(prev, channel), CBAM3D(channel)))
                prev = channel
            self.blocks = nn.ModuleList(blocks)

        def forward(self, x: Any) -> list[Any]:
            """Return multi-scale encoder features from shallow to deep."""

            feats = []
            for i, block in enumerate(self.blocks):
                x = block(x)
                feats.append(x)
                if i != len(self.blocks) - 1:
                    x = F.max_pool3d(x, kernel_size=2, stride=2)
            return feats


    class AttentionDecoder(nn.Module):
        """Decoder with transposed convolutions and skip fusion."""

        def __init__(self, channels: Sequence[int] = (24, 48, 96, 192)) -> None:
            super().__init__()
            c1, c2, c3, c4 = channels
            self.up3 = nn.ConvTranspose3d(c4, c3, kernel_size=2, stride=2)
            self.conv3 = ResBlock3D(c3 + c3, c3)
            self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=2, stride=2)
            self.conv2 = ResBlock3D(c2 + c2, c2)
            self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=2, stride=2)
            self.conv1 = ResBlock3D(c1 + c1, c1)

        def forward(self, feats: Sequence[Any]) -> Any:
            """Decode encoder features into the highest-resolution feature map."""

            f1, f2, f3, f4 = feats
            x = self.up3(f4)
            x = self.conv3(torch.cat([_match_shape(x, f3), f3], dim=1))
            x = self.up2(x)
            x = self.conv2(torch.cat([_match_shape(x, f2), f2], dim=1))
            x = self.up1(x)
            x = self.conv1(torch.cat([_match_shape(x, f1), f1], dim=1))
            return x


    class AttentionFaultFormerNet(nn.Module):
        """Compact attention U-Net style 3D fault segmentation network."""

        def __init__(
            self,
            in_channels: int = 1,
            out_channels: int = 1,
            channels: Sequence[int] = (24, 48, 96, 192),
        ) -> None:
            super().__init__()
            self.encoder = AttentionFaultFormerEncoder(in_channels, channels)
            self.decoder = AttentionDecoder(channels)
            self.head = nn.Conv3d(channels[0], out_channels, kernel_size=1)

        def forward(self, x: Any) -> Any:
            """Return segmentation logits for an input volume batch."""

            return self.head(self.decoder(self.encoder(x)))


    def _match_shape(x: Any, reference: Any) -> Any:
        if x.shape[2:] == reference.shape[2:]:
            return x
        return F.interpolate(x, size=reference.shape[2:], mode="trilinear", align_corners=False)

else:

    class AttentionFaultFormerNet:  # type: ignore
        """Placeholder that raises when PyTorch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()
