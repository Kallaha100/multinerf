#!/bin/bash
set -e  # exit on error
set -x

# 1. Prepare directories
mkdir -p model/checkpoints

#2 get to model
cd model

# 3. Download/clone your data
gdown --folder https://drive.google.com/drive/folders/1WHA3v7_HuXpLCjSxCl0MW-z6S4-GkXxs?usp=sharing

#4
cd ..

# 5. Run training
python -m train \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.data_dir = 'model'" \
  --gin_bindings="Config.checkpoint_dir = 'model/checkpoints'" \
  --logtostderr


# just specify model/checkpoints as the result to retrieve
