#!/bin/sh
PREFIX=$HOME/.local
BINDIR=$PREFIX/bin
SCRIPT=$BINDIR/laslnet
mkdir -p $BINDIR
printf "#!/usr/bin/env -S -u LD_LIBRARY_PATH %s/venv/bin/python3\n" $PWD >$SCRIPT
cat scripts/laslnet.py >>$SCRIPT
chmod +x $SCRIPT
