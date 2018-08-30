#!/usr/bin/env bash
git reset --hard
git pull
sleep 3
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
cd ${CURRENT_DIR}
python setup.py install --user
cd -