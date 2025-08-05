#!/bin/bash
set -e
set -x
# 1. Prepare directories
mkdir -p model/checkpoints
#2 get to model
cd model
# 3. Download/clone your data
gdown "https://drive.google.com/uc?id=1wBh3w32F7aV5bWWXUd6eMBaLB5MCNJyd"
unzip nerf.zip
rm nerf.zip
#4
cd ..
# 5. Run training
python -m train \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = 'model'" \
  --gin_bindings="Config.checkpoint_dir = 'model/checkpoints'" \
  --logtostderr
# just specify model/checkpoints as the result to retrieve

