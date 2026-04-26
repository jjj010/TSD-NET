#!/usr/bin/env bash
set -e
python scripts/create_dummy_data.py --out prepare_data/data/SMOKE_data.csv --n 80
python main.py --datafile SMOKE --input-len 4 --horizon 1 --feature-set 3 --num-blocks 1 --epochs 1 --batch-size 64 --no-profiler
python scripts/profile_complexity.py --input-size 3 --input-len 4 --horizon 1 --num-blocks 1 --runs 1 --channels 4 --out results/smoke_complexity.csv
