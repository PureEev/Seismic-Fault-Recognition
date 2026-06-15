# Models and Loss Functions

This document provides a detailed overview of the neural network architectures and loss functions implemented in the `seismic_fault_recognition` package.

---

## 🏗 Model Architectures

### 1. Swin Tiny
The project uses one checkpoint-compatible `swin_tiny` architecture. It accepts
only `128 x 128 x 128` inputs, uses a patch-5 projection, feature widths
`48 -> 96 -> 192`, and removes the deepest MONAI SwinUNETR stage.

### 2. OmniSeis (Super-Resolution)
OmniSeis is a GAN-based architecture for 3D seismic super-resolution.
- **Generator:** Based on an encoder-decoder structure with residual blocks to capture multi-scale features.
- **Discriminator:** A **PatchGAN 3D** discriminator that penalizes high-frequency structure at the scale of local patches.

### 3. FaultFormer
An experimental architecture that leverages global attention mechanisms to better capture the connectivity of faults across large 3D volumes.

---

## 📉 Loss Functions

We use composite loss functions to balance voxel-wise accuracy with structural connectivity.

### 1. Segmentation Losses
- **BCEWithLogitsLoss:** The baseline for binary classification of voxels.
- **Dice Loss:** Handles class imbalance by optimizing the overlap between predicted and ground truth masks.
- **Focal Loss:** Focuses the model on "hard" examples (thin faults or noisy areas) by down-weighting well-classified voxels.
- **SymCombinedLoss:** A weighted sum of Dice, Focal, and BCE losses.

### 2. Super-Resolution Losses
- **L1 Loss:** Ensures pixel-level fidelity.
- **VGG Perceptual Loss:** Uses a pretrained 2D VGG network (applied slice-wise) to ensure the generated cubes are "perceptually" similar to real seismic data.
- **Adversarial Loss:** Driven by the PatchGAN discriminator to ensure the generator produces realistic textures.

### 3. SimMIM (Pretraining) Loss
- **Masked L1 Loss:** Reconstruction loss calculated **only** on the masked patches. This forces the model to learn the underlying spatial correlations of seismic data.

---

## 🧪 Training Profiles

Profiles in `src/seismic_fault_recognition/trainers.py` define the specific training loops:
- `faultseg3d_pretrain`: Standard supervised loop with AMP.
- `thebe_finetune`: Optimized loop with gradient accumulation for high-resolution cubes.
- `sr_training`: GAN-based loop with alternating generator/discriminator updates.
