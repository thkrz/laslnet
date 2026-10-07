#!/bin/sh
mkdir -p bin
printf "#!/usr/bin/env -S -u LD_LIBRARY_PATH %s/venv/bin/python3\n" $PWD > bin/laslnet
cat scripts/laslnet.py >> bin/laslnet
chmod +x bin/laslnet
