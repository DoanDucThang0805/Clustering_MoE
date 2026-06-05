#!/bin/bash

source venv/bin/activate
cd src

python -m training.kmean \
    --dataset_name "plantdoc" \
    --type_model "non_pretrain_models" \
    --backbone_type "non_pretrain_backbone" \
    --backbone_name "mobilenetv3small_torchvision" \
    --seed 42 \
    --num_clusters 2 3 4 5 6 8
