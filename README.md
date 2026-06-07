# Clone Project

```bash
git clone https://github.com/DoanDucThang0805/Clustering_MoE.git
cd Clustering_MoE
```

# System Requirements

## Python Version
- **Python 3.11** (required)

## Operating System
- Linux (Ubuntu 18.04+)
- macOS (10.14+)
- Windows (WSL2 recommended)

## Hardware
- CPU: 8+ cores
- RAM: 16GB minimum, 32GB+ recommended
- GPU: NVIDIA GPU with CUDA Compute Capability 7.0+ (optional, for acceleration)
  - CUDA: 11.8 or higher
  - cuDNN: 8.x
- Storage: 50GB+ free space

## Python Dependencies
```
pandas
numpy
scikit-learn
matplotlib
seaborn
torch
torchvision
torchinfo
transformers
timm
albumentations
opencv-python
scipy
pillow
umap-learn
onnxscript
thop
onnxruntime
psutil
memory_profiler
```

## Installation Guide

1. Create virtual environment:
```bash
python3.11 -m venv venv
```

2. Activate environment:
```bash
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Download Dataset

```bash
mkdir -p data
curl -L -o data/tomato-plantdoc-mod.zip \
  https://www.kaggle.com/api/v1/datasets/download/cthngon/tomato-plantdoc-mod
unzip -q data/tomato-plantdoc-mod.zip -d data/
```

## Verification
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```
