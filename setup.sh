#!/usr/bin/env bash
# myslurmstatus installer.
# Copies myslurmstatus.py into ~/.local/bin (or $PREFIX/bin) and exposes it
# as `myslurmstatus`. Pure stdlib — depends only on Python 3.9+ and the
# Slurm client commands (`squeue`, `sinfo`, `scontrol`, `sacctmgr`).

set -euo pipefail

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PREFIX:-$HOME/.local}"
BIN_DIR="$PREFIX/bin"

mkdir -p "$BIN_DIR"
install -m 755 "$SRC_DIR/myslurmstatus.py" "$BIN_DIR/myslurmstatus"

echo "Installed: $BIN_DIR/myslurmstatus"

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        echo
        echo "Note: $BIN_DIR is not on your PATH."
        echo "Add this line to your ~/.bashrc or ~/.zshrc:"
        echo "  export PATH=\"$BIN_DIR:\$PATH\""
        ;;
esac

echo
echo "Run 'myslurmstatus' to launch the dashboard."
echo "Press 's' inside to pick favorites; everything else is auto-discovered."
