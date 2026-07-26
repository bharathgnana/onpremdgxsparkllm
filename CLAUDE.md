# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository manages a self-contained on-prem LLM serving stack for NVIDIA DGX Spark (Blackwell GB10). It runs as an independent `docker compose` project and exposes a single network surface that any client application can consume.

The stack consists of:
- **vLLM** (on port 8000) - primary LLM for chat/coding/agent traffic with continuous batching, prefix caching, and FP8 KV cache
- **Ollama** (on port 11434) - auxiliary service for embeddings and vision OCR fallback

## Key Components

### Architecture
- **Primary Model**: Qwen3-Coder-30B-A3B-Instruct-FP8 (~30 GB) served by vLLM
- **Auxiliary Model**: mxbai-embed-large (1024-dim) served by Ollama
- **Hybrid Setup**: Qwen3-Coder for interactive coding and GLM-5.2 for asynchronous verification

### Repository Structure
```
.
├── README.md              # Main documentation
├── docker-compose.yml     # Docker Compose configuration for vLLM + Ollama
├── Dockerfile.vllm-patch  # Patched vLLM image for sm_121 compatibility
├── .env.example           # Template for environment variables
├── models/                # Model weights (gitignored)
│   ├── qwen3-coder-30b-a3b-instruct-fp8/  # Active model
│   └── glm52-colibri-int4/                # GLM-5.2 model for hybrid setup
├── ollama-data/           # Ollama blob store (gitignored)
├── scripts/               # Utility scripts
│   ├── start.sh           # Start the stack
│   ├── status.sh          # Check status of services
│   ├── migrate_to_vllm.sh # Lifecycle management for vLLM stack
│   └── claude-hybrid.sh   # Launch Claude with hybrid setup
└── docs/                  # Documentation
    ├── ARCHITECTURE.md    # Technical architecture details
    └── VLLM_MIGRATION.md  # VLLM setup and tuning guide
```

## Common Development Tasks

### Starting the Stack
```bash
cp .env.example .env
# Edit HF_TOKEN in .env (needed only on first model download)
./scripts/start.sh         # docker compose up -d
./scripts/status.sh        # check what's running and which models are loaded
```

### Working with the Hybrid Setup (Qwen + GLM)
```bash
# Download and prepare models
./scripts/migrate_to_vllm.sh download
./scripts/download_glm52_colibri.sh

# Start hybrid services
./scripts/start-hybrid.sh

# Launch Claude with hybrid configuration
./scripts/claude-hybrid.sh
```

### Monitoring and Validation
```bash
# Check service status
./scripts/status.sh

# Validate vLLM is working
./scripts/migrate_to_vllm.sh validate

# Benchmark performance
./scripts/bench-edge.py --lane both --quick
```

### Environment Variables
Key environment variables include:
- `HF_TOKEN` - Required for downloading models from Hugging Face
- `VLLM_BIND` - Port binding for vLLM (default: 0.0.0.0)
- `OLLAMA_BIND` - Port binding for Ollama (default: 0.0.0.0)
- `VLLM_MODEL_PATH` - Path to the model directory
- `VLLM_MODEL` - Name of the served model

### Client Configuration
Launch Claude Code through the verified compatibility path:

```bash
./scripts/claude-hybrid.sh
```

Claude Code uses port 8002; OpenAI-shaped clients continue to use port 8000.

## Technical Details

### Why vLLM with sm_121 Patching
The DGX Spark runs on NVIDIA driver 580.x which lacks sm_121 support in upstream vLLM. The patched image in `Dockerfile.vllm-patch` addresses this by:
1. Upgrading transformers/tokenizers to support Qwen3 architectures
2. Forcing FP8 GEMM to use Triton path instead of CUTLASS
3. Using `--enforce-eager` to avoid CUDA graph capture crashes

### Hybrid Architecture
The hybrid setup combines:
- **Qwen3-Coder** on port 8000, reached by Claude Code through proxy port 8002
- **GLM-5.2** on port 8001 for asynchronous verification
- **Anthropic vLLM Proxy** on port 8002 to normalize requests for compatibility

### Performance Characteristics
- Qwen3-Coder: 20k-token TTFT ≤ 30 seconds, decode ≥ 15 tok/s
- GLM-5.2: measured 2k TTFT 213.452 seconds and 0.831 tok/s; asynchronous only
- Memory constraints: 119 GiB unified memory with 10+ GiB available for optimal operation