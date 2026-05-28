source venv/bin/activate
cd src

python -m inference.non_pretrain_models.inference \
    --dataset_name "plantdoc" \
    --type_model "non_pretrain_models" \
    --model_name "mobilenetv3small_timm" \
    --seed 42 \
    --run_time "run_20260528-163241"

python -m inference.non_pretrain_models.inference \
    --dataset_name "plantdoc" \
    --type_model "non_pretrain_models" \
    --model_name "mobilenetv3small_timm" \
    --seed 43 \
    --run_time "run_20260528-163944"

python -m inference.non_pretrain_models.inference \
    --dataset_name "plantdoc" \
    --type_model "non_pretrain_models" \
    --model_name "mobilenetv3small_timm" \
    --seed 44 \
    --run_time "run_20260528-163957"

python -m inference.non_pretrain_models.inference \
    --dataset_name "plantdoc" \
    --type_model "non_pretrain_models" \
    --model_name "mobilenetv3small_timm" \
    --seed 45 \
    --run_time "run_20260528-164015"

python -m inference.non_pretrain_models.inference \
    --dataset_name "plantdoc" \
    --type_model "non_pretrain_models" \
    --model_name "mobilenetv3small_timm" \
    --seed 46 \
    --run_time "run_20260528-164030"
