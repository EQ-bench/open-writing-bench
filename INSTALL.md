# Installation

## Prerequisites

Install build tools:

```bash
# Ubuntu/Debian
apt install build-essential cmake ninja-build

# For CUDA builds
apt install nvidia-cuda-toolkit
```

## Quick Start

```bash
# 1. Install base dependencies
uv sync

# 2. Activate environment and set build parallelism
source .venv/bin/activate
export MAX_JOBS=12
export CMAKE_BUILD_PARALLEL_LEVEL=12

# 3. Install GPU backends (choose what you need)

# vLLM (recommended for high-throughput GPU inference)
python scripts/install_vllm.py

# llama.cpp (for GGUF models)
python scripts/install_llama_cpp.py

# Flash Attention (optional, improves transformer performance)
uv pip install flash-attn
```

## Backend Installation Details

### vLLM

For high-throughput GPU inference with continuous batching:

```bash
python scripts/install_vllm.py
```

Options:
- `--version 0.6.0` - specific version
- `--force-source` - build from source
- `--dry-run` - preview commands
- `-v` - verbose output

### llama.cpp

For running GGUF quantized models:

```bash
python scripts/install_llama_cpp.py
```

Options:
- `--backend cuda|metal|rocm|cpu` - force specific backend
- `--version 0.3.16` - specific version
- `--allow-unofficial-wheels` - try community wheels
- `--dry-run` - preview commands

### Flash Attention

Improves attention performance for transformers backend:

```bash
uv pip install flash-attn
```

## Environment Variables

For HTTP backends:
```bash
export TEST_API_KEY="your-api-key"
export TEST_API_URL="https://api.example.com/v1/chat/completions"
```

## Verifying Installation

```bash
# Check backends are available
python -c "from utils.inference import get_backend; print('OK')"

# Test vLLM (if installed)
python -c "import vllm; print(f'vLLM {vllm.__version__}')"

# Test llama.cpp (if installed)
python -c "import llama_cpp; print(f'llama-cpp-python {llama_cpp.__version__}')"
```

## Troubleshooting

### CUDA Version Mismatch

Check versions match:
```bash
nvidia-smi                                          # Driver CUDA version
nvcc --version                                      # Toolkit version
python -c "import torch; print(torch.version.cuda)" # PyTorch CUDA
```

### Out of Memory During Build

Reduce parallel jobs further:
```bash
export MAX_JOBS=2
export CMAKE_BUILD_PARALLEL_LEVEL=2
```

### Slow Builds

Ensure ninja is installed - it's much faster than make:
```bash
apt install ninja-build
```
