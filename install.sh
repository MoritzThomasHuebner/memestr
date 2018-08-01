#!/usr/bin/env bash
git pull

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd ${CURRENT_DIR}
python setup.py install --user
cd -