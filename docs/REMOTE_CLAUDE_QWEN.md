# Use the local Qwen model with Claude Code from another laptop

This guide connects Claude Code on another laptop to the
`qwen3-coder-30b` model running on the DGX Spark.

The recommended connection uses an SSH tunnel. This keeps the Claude
compatibility proxy private and avoids exposing an unauthenticated HTTP
endpoint to the Wi-Fi network.

## 1. Start the hybrid stack on the DGX Spark

On the Spark:

```bash
cd ~/demos/onpremdgxsparkllm
./scripts/start-hybrid.sh
./scripts/hybrid-status.sh
```

The status output should report:

- Qwen on port `8000` as healthy.
- GLM on port `8001` as healthy.
- The review worker as active.

The Claude compatibility proxy starts automatically with
`./scripts/start-hybrid.sh` and is also checked by
`./scripts/claude-hybrid.sh`. It listens on the Spark.s loopback port `8002`.

The Spark's LAN address at the time this guide was written was:

```text
192.168.1.49
```

If that address changes, find the current address on the Spark:

```bash
hostname -I
```

Use the address beginning with `192.168.` or `10.`.

## 2. Open an SSH tunnel from the laptop

On the other laptop, run:

```bash
ssh -N -L 8002:127.0.0.1:8002 test1@192.168.1.49
```

Keep this terminal open while using Claude Code.

If the Spark's LAN address has changed, replace `192.168.1.49` with its
current address.

Verify the tunnel from a second laptop terminal:

```bash
curl http://127.0.0.1:8002/health
```

A successful request may return an empty response with HTTP status 200.
To display the status explicitly:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8002/health
```

Expected output:

```text
200
```

## 3. Create a Claude launcher on the laptop

Create `~/claude-qwen.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

export ANTHROPIC_BASE_URL="http://127.0.0.1:8002"
export ANTHROPIC_AUTH_TOKEN="local-qwen"
unset ANTHROPIC_API_KEY

export ANTHROPIC_MODEL="qwen3-coder-30b"
export ANTHROPIC_DEFAULT_OPUS_MODEL="qwen3-coder-30b"
export ANTHROPIC_DEFAULT_SONNET_MODEL="qwen3-coder-30b"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="qwen3-coder-30b"

exec claude "$@"
```

Make it executable:

```bash
chmod +x ~/claude-qwen.sh
```

Launch Claude from the project directory you want it to work on:

```bash
cd /path/to/your/project
~/claude-qwen.sh
```

Claude should display:

```text
qwen3-coder-30b · API Usage Billing
```

Test the interactive integration:

```text
/init
```

## 4. Optional persistent SSH configuration

Add an entry to `~/.ssh/config` on the laptop:

```sshconfig
Host dgx-spark
    HostName 192.168.1.49
    User test1
    LocalForward 8002 127.0.0.1:8002
    ServerAliveInterval 30
    ServerAliveCountMax 3
    ExitOnForwardFailure yes
```

The tunnel can then be started with:

```bash
ssh -N dgx-spark
```

## Troubleshooting

### Port 8002 is unreachable

Confirm the tunnel is running on the laptop:

```bash
curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8002/health
```

On the Spark, start the compatibility proxy if needed:

```bash
cd ~/demos/onpremdgxsparkllm
./scripts/anthropic-vllm-proxy.py --ensure
```

Then confirm both services are listening:

```bash
ss -ltn | grep -E ':(8000|8002)\b'
```

### Claude reports a system-message validation error

Make sure the laptop uses port `8002`, not direct vLLM port `8000`:

```bash
echo "$ANTHROPIC_BASE_URL"
```

Expected:

```text
http://127.0.0.1:8002
```

Close any older Claude process and relaunch it through `~/claude-qwen.sh`.
Existing Claude processes do not inherit updated environment variables.

### Claude warns that both authentication variables are set

Use `ANTHROPIC_AUTH_TOKEN` only:

```bash
unset ANTHROPIC_API_KEY
export ANTHROPIC_AUTH_TOKEN="local-qwen"
```

The launcher above performs this automatically.

### The SSH connection drops during a session

Restart the tunnel:

```bash
ssh -N dgx-spark
```

Claude requests will fail while the tunnel is disconnected, but the model
continues running on the Spark.

## Security notes

- Do not bind port `8002` directly to `0.0.0.0`; the proxy does not provide
  independent authentication.
- SSH encrypts the laptop-to-Spark connection and limits access to users who
  can authenticate to the Spark.
- The laptop does not need model weights. It only needs Claude Code and an SSH
  client.
- Port `8002` must be used for Claude Code because it normalizes hook-injected
  Anthropic system messages before forwarding requests to vLLM on port `8000`.
