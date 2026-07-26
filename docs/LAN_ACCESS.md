# Use GLM-5.2 from another laptop on the same Wi‑Fi

Colibrì does **not** use a username/password form. It uses a single **API key**
(sent as `Authorization: Bearer …` or `x-api-key`). Treat `COLI_API_KEY` as your
shared secret.

## 1. On the DGX Spark (server)

Edit `/home/test1/demos/onpremdgxsparkllm/.env`:

```bash
COLIBRI_BIND=0.0.0.0
COLIBRI_PORT=8001
COLI_API_KEY=replace-with-a-long-random-secret
```

Generate a secret (example):

```bash
openssl rand -hex 32
```

Restart the service:

```bash
systemctl --user daemon-reload
systemctl --user restart colibri-glm52.service
journalctl --user -u colibri-glm52.service -f
```

Find the Spark’s LAN address (use the `192.168.x.x` or `10.x.x.x` Wi‑Fi IP):

```bash
hostname -I | awk '{print $1}'
```

### Firewall

If `ufw` is active, allow your home LAN only (adjust subnet):

```bash
sudo ufw allow from 192.168.0.0/16 to any port 8001 proto tcp comment 'colibri-glm52'
sudo ufw status
```

### Smoke test from the Spark

```bash
KEY='your-secret-from-env'
curl -s http://127.0.0.1:8001/v1/models -H "Authorization: Bearer $KEY"
```

### Smoke test from the other laptop

```bash
KEY='your-secret-from-env'
SPARK=192.168.1.50   # Spark Wi‑Fi IP
curl -s "http://${SPARK}:8001/v1/models" -H "Authorization: Bearer $KEY"
```

Wrong or missing key → HTTP **401**.

## 2. Claude Code on the other laptop

In `~/.claude/settings.json`:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "http://192.168.1.50:8001",
    "ANTHROPIC_AUTH_TOKEN": "your-secret-from-env",
    "ANTHROPIC_MODEL": "glm-5.2-colibri",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-5.2-colibri",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-5.2-colibri",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-5.2-colibri"
  }
}
```

Replace `192.168.1.50` and the key with your real values. Use `ANTHROPIC_AUTH_TOKEN` only, then run `claude`.

## 3. OpenAI-shaped clients (Cursor, aider, etc.)

- Base URL: `http://192.168.1.50:8001/v1`
- API key: same as `COLI_API_KEY`
- Model: `glm-5.2-colibri`

## Security notes

- Use a **strong random** `COLI_API_KEY`; anyone on the LAN who has it can use the model.
- Prefer **Wi‑Fi you trust**; this is not TLS. For untrusted networks, use SSH tunnel
  instead of `0.0.0.0`:

  ```bash
  # On laptop:
  ssh -N -L 8001:127.0.0.1:8001 test1@192.168.1.50
  # Then ANTHROPIC_BASE_URL=http://127.0.0.1:8001
  ```

- Only one generation runs at a time; a second client **queues** (by design).

## Boot at startup

```bash
sudo loginctl enable-linger test1
```

Service must be enabled: `./scripts/install-colibri-service.sh`
