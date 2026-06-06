#!/bin/bash

source venv/bin/activate
cd src
python -m training.kmean \
    --dataset_name "plantdoc" \
    --backbone_type "non_pretrain_backbone" \
    --backbone_name "mobilenetv3small_torchvision" \
    --seed 42 \
    --num_clusters 2 3 4 5 6 8 \
    --metric "cosine"

python -m training.kmean \
    --dataset_name "plantdoc" \
    --backbone_type "non_pretrain_backbone" \
    --backbone_name "mobilenetv3small_torchvision" \
    --seed 43 \
    --num_clusters 2 3 4 5 6 8 \
    --metric "cosine"

python -m training.kmean \
    --dataset_name "plantdoc" \
    --backbone_type "non_pretrain_backbone" \
    --backbone_name "mobilenetv3small_torchvision" \
    --seed 44 \
    --num_clusters 2 3 4 5 6 8 \
    --metric "cosine"

python -m training.kmean \
    --dataset_name "plantdoc" \
    --backbone_type "non_pretrain_backbone" \
    --backbone_name "mobilenetv3small_torchvision" \
    --seed 45 \
    --num_clusters 2 3 4 5 6 8 \
    --metric "cosine"

python -m training.kmean \
    --dataset_name "plantdoc" \
    --backbone_type "non_pretrain_backbone" \
    --backbone_name "mobilenetv3small_torchvision" \
    --seed 46 \
    --num_clusters 2 3 4 5 6 8 \
    --metric "cosine"
