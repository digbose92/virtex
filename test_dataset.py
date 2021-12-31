from virtex.config import Config
from virtex.factories import (
    PretrainingDatasetFactory, PretrainingModelFactory, OptimizerFactory,
    LRSchedulerFactory,
)
from torch.utils.data import Dataset, DataLoader

_C = Config("configs/_base_bicaptioning_R_50_L1_H1024.yaml")
train_dataset = PretrainingDatasetFactory.from_config(_C, split="train")
artemis_dl=DataLoader(train_dataset,batch_size=1,shuffle=False)
dict_artemis_data=next(iter(artemis_dl))

CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/pretrain_virtex.py \
    --config configs/_base_bicaptioning_R_50_L1_H1024.yaml \
    --num-machines 1 \
    --num-gpus-per-machine 8 \
    --cpu-workers 4 \
    --serialization-dir /tmp/VIRTEX_R_50_L1_H1024
