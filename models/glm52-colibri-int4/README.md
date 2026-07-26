---
license: mit
base_model: zai-org/GLM-5.2-FP8
base_model_relation: quantized
language:
  - en
  - zh
tags:
  - glm_moe_dsa
  - int4
  - cpu
  - moe
  - expert-streaming
  - colibri
---

# GLM-5.2 — colibrì int4 container (~370 GB)

`This is the *EXACT SAME THING* as https://huggingface.co/jlnsrk/GLM-5.2-colibri-int4, BUT with int8 MTP heads, which are needed for speculative decoding—and with that, an overall major inference speedboost. The original int4 MTP heads have low acceptance rate, and are essentially useless.`

**⚠️ This is NOT a GGUF / AWQ / GPTQ / MLX model.** It only works with the colibrì engine.

## Usage

```bash
# Get the engine
git clone https://github.com/JustVugg/colibri && cd colibri/c && ./setup.sh

# download this repo to a FAST local disk
hf download mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp --local-dir /nvme/glm52

# run on CPU
COLI_MODEL=/nvme/glm52 ./coli chat
```

Requirements: Linux (or WSL2), gcc + OpenMP, AVX2, ≥16 GB RAM, ~400 GB free NVMe.

## Provenance & license

Converted from [zai-org/GLM-5.2-FP8](https://huggingface.co/zai-org/GLM-5.2-FP8) (MIT). This derivative is likewise MIT. Conversion performed with colibrì's official converter, unmodified.
Cloned & modded from [jlnsrk/GLM-5.2-colibri-int4](https://huggingface.co/jlnsrk/GLM-5.2-colibri-int4)
