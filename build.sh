#!/bin/bash
dldir="bin"

[ ! -d "$dldir" ] && mkdir -p "$dldir"

cd "$dldir"
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j

