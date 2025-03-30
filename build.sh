#!/bin/bash
dl_dir="build"

[ ! -d "$dl_dir" ] && mkdir -p "$dl_dir"

# shellcheck disable=SC2164
cd "$dl_dir"
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j