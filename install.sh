#!/usr/bin/env bash
git pull

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
INSTALL_SCRIPT="$CURRENT_DIR/setup.py"

python ${INSTALL_SCRIPT} install --user