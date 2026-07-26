# Models

Weights live here (gitignored). The vLLM Qwen stack was removed for the
document-intelligence PoC wind-down; **GLM-5.2** is served by
[Colibrì](https://github.com/JustVugg/colibri) on the host (not via
`docker-compose`).

## Active model (Colibrì)

| Directory | Hugging Face ID | Size | Runtime |
|---|---|---|---|
| `glm52-colibri-int4/` | [`mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp`](https://huggingface.co/mateogrgic/GLM-5.2-colibri-int4-with-int8-mtp) | ~372 GB | `./scripts/start-colibri.sh` → OpenAI API on port **8001** |

Use the **int8 MTP** variant only (int4 MTP heads break speculative decoding).

## Download

```bash
# Foreground (resumable):
./scripts/download_glm52_colibri.sh

# Background with log:
nohup ./scripts/download_glm52_colibri.sh >> models/glm52-colibri-int4/.download.log 2>&1 &
tail -f models/glm52-colibri-int4/.download.log
```

Requires `HF_TOKEN` in `.env`. Progress also visible under
`models/glm52-colibri-int4/.cache/` while downloading.

## After download

```bash
COLI_MODEL=$PWD/models/glm52-colibri-int4 /home/test1/demos/colibri/c/coli doctor
./scripts/start-colibri.sh
```

Stop the old compose stack before serving GLM (128 GB UMA is shared):

```bash
docker compose stop    # or: docker compose down
```
