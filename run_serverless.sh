#!/bin/bash
set -e
set -x
# 1. Prepare directories
mkdir -p model
#2 get to model
cd model
# 3. Download/clone your data
gdown --id 1lWld8MCUf4tGdq6BBiskzeAglcXOai0T -O nerf.zip
unzip nerf.zip
rm nerf.zip
#4
cd ..
# 5. Run training
python -m train \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = 'model'" \
  --gin_bindings="Config.checkpoint_dir = '/outputs/checkpoints'" \
  --gin_bindings="Config.max_steps = 5000" \
  --logtostderr | tee /outputs/train.log
# just specify /output/checkpoints in runpod as the result to retrieve
#This file is an addition to run the training automatically






