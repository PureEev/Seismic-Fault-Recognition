# Final Swin Tiny

The project exposes one Swin Tiny model under the registry key `swin_tiny`.
All training, validation, CLI, and notebook workflows use this architecture.

Fixed contract:

- input spatial size: `128 x 128 x 128`;
- input/output channels: `1 / 1` by default;
- feature widths: `48 -> 96 -> 192`;
- patch projection: `Conv3d(kernel_size=5, stride=2, padding=2)`;
- deepest Swin stage, `encoder10`, and `decoder5` are removed;
- decoder path: `decoder3 -> decoder2 -> decoder1`.

```python
from seismic_fault_recognition.models.factory import build_model_by_name

model = build_model_by_name(
    "swin_tiny",
    img_size=(128, 128, 128),
    in_channels=1,
    out_channels=1,
)
```

Any other configured or runtime spatial size raises `ValueError`. The model is
compatible with `swinunetr_tiny_after_faultseg3d.pth` after optional removal of
a leading `module.` prefix.

Checkpoint diagnostics remain available through:

```bash
sfr checkpoint inspect checkpoint.pth --json
```
