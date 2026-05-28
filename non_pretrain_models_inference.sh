source venv/bin/activate
cd src

python -m inference.non_pretrain_models.inference \
    --dataset_name "plantdoc" \
    --type_model "non_pretrain_models" \
    --model_name "mobilenetv3small_torchvision" \
    --seed 43 \
    --run_time "run_20260528-140140"

python -m inference.non_pretrain_models.inference \
    --dataset_name "plantdoc" \
    --type_model "non_pretrain_models" \
    --model_name "mobilenetv3small_torchvision" \
    --seed 44 \
    --run_time "run_20260528-140206"

python -m inference.non_pretrain_models.inference \
    --dataset_name "plantdoc" \
    --type_model "non_pretrain_models" \
    --model_name "mobilenetv3small_torchvision" \
    --seed 45 \
    --run_time "run_20260528-140218"

python -m inference.non_pretrain_models.inference \
    --dataset_name "plantdoc" \
    --type_model "non_pretrain_models" \
    --model_name "mobilenetv3small_torchvision" \
    --seed 46 \
    --run_time "run_20260528-140233"