#!/usr/bin/env bash
# Install Colibrì as a systemd *user* service (starts at login; use linger for boot).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_SRC="${ROOT}/deploy/systemd/colibri-glm52.service"
UNIT_DST="${HOME}/.config/systemd/user/colibri-glm52.service"

mkdir -p "${HOME}/.config/systemd/user"
cp "${UNIT_SRC}" "${UNIT_DST}"

systemctl --user daemon-reload
systemctl --user enable colibri-glm52.service

echo ""
echo "Installed: ${UNIT_DST}"
echo "Enabled:   colibri-glm52.service (user)"
echo ""
echo "Start now:"
echo "  systemctl --user start colibri-glm52.service"
echo "  systemctl --user status colibri-glm52.service"
echo "  journalctl --user -u colibri-glm52.service -f"
echo ""
if loginctl show-user "$(id -un)" -p Linger 2>/dev/null | grep -q 'Linger=no'; then
  echo "Boot without login: run once as root:"
  echo "  sudo loginctl enable-linger $(id -un)"
  echo "Then reboot — the service starts at boot (model load ~3–10 min)."
else
  echo "Linger is enabled — service will start at boot."
fi
echo ""
echo "Health: curl -s http://127.0.0.1:8001/health | head"
echo "Models: curl -s http://127.0.0.1:8001/v1/models"
