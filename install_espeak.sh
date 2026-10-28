#!/usr/bin/env bash
# Install espeak-ng using the platform's package manager.
set -euo pipefail

case "$(uname -s)" in
  Darwin)
    if ! command -v brew >/dev/null 2>&1; then
      echo "Homebrew not found. Install it from https://brew.sh and re-run." >&2
      exit 1
    fi
    brew install espeak-ng
    ;;
  Linux)
    SUDO=""
    if [ "$(id -u)" -ne 0 ]; then
      SUDO="sudo"
    fi
    $SUDO apt-get update -qq
    $SUDO apt-get install -qq -y espeak-ng
    ;;
  *)
    echo "Unsupported OS: $(uname -s)" >&2
    exit 1
    ;;
esac

espeak-ng --version
