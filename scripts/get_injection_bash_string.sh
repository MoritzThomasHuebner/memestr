#!/usr/bin/env bash
FILENAME=$1
PARAMS=`python -c "import memestr; print(memestr.submit.submitter.find_unallocated_name(name=\"$FILENAME\"))"`