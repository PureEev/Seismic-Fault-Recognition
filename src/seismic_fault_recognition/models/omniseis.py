"""Reusable OmniSeis and super-resolution building blocks."""

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
        raise ImportError("PyTorch is required for OmniSeis models")


from seismic_fault_recognition.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register("omniseis_segmentation")
@MODEL_REGISTRY.register("omniseis_encoder")
def build_omniseis_segmentation(**kwargs: Any) -> Any:
    """Build the OmniSeis segmentation model."""
    from .swinunetr_variants import SWIN_TINY_FEATURE_SIZE, build_swinunetr_tiny

    backbone = build_swinunetr_tiny(**kwargs)
    encoder = OmniSeisEncoder(
        backbone,
        channels=(
            SWIN_TINY_FEATURE_SIZE,
            SWIN_TINY_FEATURE_SIZE * 2,
            SWIN_TINY_FEATURE_SIZE * 4,
        ),
        embed_dim=SWIN_TINY_FEATURE_SIZE * 4,
    )
    return OmniSeisSegmentationModel(encoder)

@MODEL_REGISTRY.register("omniseis_sr_generator")
def build_omniseis_sr_generator(**kwargs: Any) -> Any:
    """Build the OmniSeis SR generator model."""
    return OmniSeisSRGenerator(**kwargs)

@MODEL_REGISTRY.register("sr_patchgan_discriminator")
@MODEL_REGISTRY.register("patchgan_discriminator_3d")
def build_patchgan_discriminator_3d(**kwargs: Any) -> Any:
    """Build the 3D PatchGAN discriminator model."""
    return PatchGANDiscriminator3D(**kwargs)

if torch is not None:

    class SafeNorm3d(nn.Module):
        """InstanceNorm3d that skips degenerate spatial tensors."""

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.norm = nn.InstanceNorm3d(channels, affine=True)

        def forward(self, x: Any) -> Any:
            """Normalize only when every spatial axis has size greater than one."""

            if min(x.shape[2:]) <= 1:
                return x
            return self.norm(x)


    class SABBlock3D(nn.Module):
        """Spatial aggregation block for 3D OmniSeis features."""

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv3d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
                SafeNorm3d(channels),
                nn.GELU(),
                nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            )

        def forward(self, x: Any) -> Any:
            """Return residual spatially aggregated features."""

            return x + self.block(x)


    class FABBlock3D(nn.Module):
        """Frequency aggregation block using FFT amplitude features."""

        def __init__(self, channels: int) -> None:
            super().__init__()
            self.proj = nn.Sequential(nn.Conv3d(channels, channels, kernel_size=1), nn.GELU())

        def forward(self, x: Any) -> Any:
            """Return residual frequency-enhanced features."""

            freq = torch.fft.rfftn(x.float(), dim=(-3, -2, -1), norm="ortho")
            amp = torch.fft.irfftn(freq.abs(), s=x.shape[-3:], dim=(-3, -2, -1), norm="ortho")
            return x + self.proj(amp.to(dtype=x.dtype))


    class TemporalAttentionBlock3D(nn.Module):
        """Depth/time-axis multi-head attention block for 3D tensors."""

        def __init__(self, channels: int, num_heads: int = 4) -> None:
            super().__init__()
            self.channels = channels
            self.norm = nn.LayerNorm(channels)
            self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

        def forward(self, x: Any) -> Any:
            """Apply attention over the depth axis for each spatial trace."""

            b, c, d, h, w = x.shape
            seq = x.permute(0, 3, 4, 2, 1).reshape(b * h * w, d, c)
            attended, _ = self.attn(self.norm(seq), self.norm(seq), self.norm(seq), need_weights=False)
            out = (seq + attended).reshape(b, h, w, d, c).permute(0, 4, 3, 1, 2)
            return out


    class ASPPBlock3D(nn.Module):
        """3D atrous spatial pyramid block with residual output."""

        def __init__(self, channels: int, dilations: Sequence[int] = (1, 2, 4)) -> None:
            super().__init__()
            branches = [
                nn.Conv3d(channels, channels, kernel_size=1, bias=False),
                *[
                    nn.Conv3d(channels, channels, kernel_size=3, padding=d, dilation=d, bias=False)
                    for d in dilations
                ],
            ]
            self.branches = nn.ModuleList(branches)
            self.out = nn.Sequential(
                nn.Conv3d(channels * len(branches), channels, kernel_size=1, bias=False),
                SafeNorm3d(channels),
                nn.GELU(),
            )

        def forward(self, x: Any) -> Any:
            """Apply parallel dilated convolutions and residual fusion."""

            return x + self.out(torch.cat([branch(x) for branch in self.branches], dim=1))


    class CascadedDomainAggregator(nn.Module):
        """Aggregate three feature levels with a domain-specific block factory."""

        def __init__(self, channels: Sequence[int], block_factory: Any) -> None:
            super().__init__()
            c1, c2, c3 = channels
            self.block1 = block_factory(c1)
            self.block2 = block_factory(c2)
            self.block3 = block_factory(c3)
            self.pool12 = nn.Sequential(nn.MaxPool3d(2), nn.Conv3d(c1, c2, kernel_size=1))
            self.pool23 = nn.Sequential(nn.MaxPool3d(2), nn.Conv3d(c2, c3, kernel_size=1))
            self.norm = nn.LayerNorm(c3)

        def forward(self, f1: Any, f2: Any, f3: Any) -> Any:
            """Return flattened contextual features for the deepest level."""

            o1 = self.block1(f1)
            o2 = self.block2(f2 + self.pool12(o1))
            o3 = self.block3(f3 + self.pool23(o2))
            b, c, d, h, w = o3.shape
            return self.norm(o3.reshape(b, c, d * h * w).transpose(1, 2))


    class FIM(nn.Module):
        """Feature interaction module for multiple domain representations."""

        def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
            super().__init__()
            self.attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
            self.norm = nn.LayerNorm(embed_dim)

        def forward(self, tensors: Sequence[Any]) -> list[Any]:
            """Fuse domains independently at each spatial token position."""

            if not tensors:
                return []
            token_count = tensors[0].shape[1]
            if any(tensor.shape[1] != token_count for tensor in tensors):
                raise ValueError("FIM domain tensors must have matching token counts")

            stacked = torch.stack(tensors, dim=2)
            batch, tokens, domains, channels = stacked.shape
            domain_sequences = stacked.reshape(batch * tokens, domains, channels)
            normalized = self.norm(domain_sequences)
            context, _ = self.attn(normalized, normalized, normalized, need_weights=False)
            fused = (domain_sequences + context).reshape(batch, tokens, domains, channels)
            return [fused[:, :, index, :] for index in range(domains)]


    class OmniSeisEncoder(nn.Module):
        """OmniSeis encoder combining spatial, frequency, temporal and context branches."""

        def __init__(
            self,
            spatial_backbone: Any,
            channels: Sequence[int] = (48, 96, 192),
            embed_dim: int = 192,
        ) -> None:
            super().__init__()
            self.spatial = spatial_backbone
            self.channels = tuple(int(channel) for channel in channels)
            self.embed_dim = int(embed_dim)
            self.spatial_agg = CascadedDomainAggregator(channels, SABBlock3D)
            self.freq_agg = CascadedDomainAggregator(channels, FABBlock3D)
            self.temp_agg = CascadedDomainAggregator(channels, TemporalAttentionBlock3D)
            self.context_agg = CascadedDomainAggregator(channels, ASPPBlock3D)
            self.fim = FIM(embed_dim)
            self.domain_weights = nn.Parameter(torch.ones(4))
            self.fusion = nn.Sequential(nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1), nn.GELU())

        def forward(self, x: Any) -> dict[str, Any]:
            """Return encoder features keyed as ``enc1`` through ``enc4``."""

            from .swinunetr_variants import SWIN_TINY_INPUT_SIZE

            spatial_size = tuple(int(value) for value in x.shape[-3:])
            if spatial_size != SWIN_TINY_INPUT_SIZE:
                raise ValueError(
                    f"OmniSeis Swin encoder requires input spatial size "
                    f"{SWIN_TINY_INPUT_SIZE}, got {spatial_size}"
                )

            enc1 = self.spatial.encoder1(x)
            enc2 = self.spatial.encoder2(F.max_pool3d(enc1, kernel_size=2, stride=2))
            enc3 = self.spatial.encoder3(self.spatial.enc2to3(F.max_pool3d(enc2, kernel_size=2, stride=2)))
            enc4 = self.spatial.encoder4(self.spatial.enc3to4(F.max_pool3d(enc3, kernel_size=2, stride=2)))

            reps = self.fim(
                [
                    self.spatial_agg(enc2, enc3, enc4),
                    self.freq_agg(enc2, enc3, enc4),
                    self.temp_agg(enc2, enc3, enc4),
                    self.context_agg(enc2, enc3, enc4),
                ]
            )
            weights = F.softmax(self.domain_weights, dim=0)
            fused = sum(rep * weight for rep, weight in zip(reps, weights))
            b, c, d, h, w = enc4.shape
            bottleneck = fused.transpose(1, 2).reshape(b, c, d, h, w)
            return {"enc1": enc1, "enc2": enc2, "enc3": enc3, "enc4": self.fusion(bottleneck)}


    class SegUpBlock3D(nn.Module):
        """Upsampling block with skip concatenation for segmentation decoders."""

        def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
            super().__init__()
            self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                nn.Conv3d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
                SafeNorm3d(out_channels),
                nn.GELU(),
                nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.GELU(),
            )

        def forward(self, x: Any, skip: Any) -> Any:
            """Upsample ``x``, align it to ``skip``, and fuse both tensors."""

            x = self.up(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            return self.conv(torch.cat([x, skip], dim=1))


    class OmniSeisSegmentationModel(nn.Module):
        """Segmentation head built on top of an OmniSeis encoder."""

        def __init__(self, encoder: OmniSeisEncoder, out_channels: int = 1) -> None:
            super().__init__()
            self.encoder = encoder
            enc2_channels, enc3_channels, enc4_channels = encoder.channels
            self.up3 = SegUpBlock3D(enc4_channels, enc3_channels, enc3_channels)
            self.up2 = SegUpBlock3D(enc3_channels, enc2_channels, enc2_channels)
            self.head = nn.Conv3d(enc2_channels, out_channels, kernel_size=1)

        def forward(self, x: Any) -> Any:
            """Return segmentation logits for one volume batch."""

            feats = self.encoder(x)
            x = self.up3(feats["enc4"], feats["enc3"])
            x = self.up2(x, feats["enc2"])
            if x.shape[2:] != feats["enc1"].shape[2:]:
                x = F.interpolate(x, size=feats["enc1"].shape[2:], mode="trilinear", align_corners=False)
            return self.head(x)


    class OmniSeisSRGenerator(nn.Module):
        """3D seismic super-resolution generator with optional encoder bottleneck."""

        def __init__(
            self,
            encoder: Any | None = None,
            in_channels: int = 1,
            base_channels: int = 32,
            scale_factor: int = 1,
        ) -> None:
            super().__init__()
            self.encoder = encoder
            self.scale_factor = scale_factor
            if encoder is not None:
                bottleneck_channels = int(getattr(encoder, "embed_dim", 96))
                self.decoder = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                    nn.Conv3d(bottleneck_channels, 96, kernel_size=3, padding=1),
                    SafeNorm3d(96),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                    nn.Conv3d(96, 48, kernel_size=3, padding=1),
                    SafeNorm3d(48),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
                    nn.Conv3d(48, 24, kernel_size=3, padding=1),
                    SafeNorm3d(24),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv3d(24, in_channels, kernel_size=3, padding=1),
                )
                self.net = None
            else:
                self.decoder = None
                self.net = nn.Sequential(
                    nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1),
                    nn.GELU(),
                    nn.Conv3d(base_channels, in_channels, kernel_size=3, padding=1),
                )

        def forward(self, x: Any) -> Any:
            """Return a super-resolved residual prediction."""

            if self.encoder is not None and self.decoder is not None:
                features = self.encoder(x)
                if isinstance(features, dict):
                    bottleneck = features["enc4"]
                elif isinstance(features, (tuple, list)):
                    bottleneck = features[-1]
                else:
                    bottleneck = features
                residual = self.decoder(bottleneck)
                if residual.shape[2:] != x.shape[2:]:
                    residual = F.interpolate(residual, size=x.shape[2:], mode="trilinear", align_corners=False)
                return x + torch.clamp(residual, min=-5.0, max=5.0)

            residual = x
            if self.scale_factor != 1:
                residual = F.interpolate(
                    residual,
                    scale_factor=self.scale_factor,
                    mode="trilinear",
                    align_corners=False,
                )
                x = residual
            if self.net is None:
                return residual
            return residual + self.net(x)


    class PatchGANDiscriminator3D(nn.Module):
        """3D PatchGAN discriminator for SR training."""

        def __init__(self, in_channels: int = 1, base_channels: int = 64) -> None:
            super().__init__()
            ndf = int(base_channels)
            self.layers = nn.Sequential(
                nn.Conv3d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
                SafeNorm3d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
                SafeNorm3d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1, bias=False),
                SafeNorm3d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv3d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
            )

        def forward(self, x: Any) -> Any:
            """Return patch-level discriminator logits."""

            return self.layers(x)

else:

    class OmniSeisEncoder:  # type: ignore
        """Placeholder that raises when PyTorch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


    class OmniSeisSegmentationModel:  # type: ignore
        """Placeholder that raises when PyTorch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


    class OmniSeisSRGenerator:  # type: ignore
        """Placeholder that raises when PyTorch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


    class PatchGANDiscriminator3D:  # type: ignore
        """Placeholder that raises when PyTorch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()
