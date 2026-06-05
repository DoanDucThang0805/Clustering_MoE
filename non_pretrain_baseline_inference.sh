source venv/bin/activate
cd src

python -m inference.non_pretrain_baseline.inference \
    --dataset_name "plantdoc" \
    --type_model "non_pretrain_baseline" \
    --model_name "mobilenetv3small_torchvision" \
    --seed 42 \
    --run_time "run_20260531-135810"

python -m inference.non_pretrain_baseline.inference \
    --dataset_name "plantdoc" \
    --type_model "non_pretrain_baseline" \
    --model_name "mobilenetv3small_torchvision" \
    --seed 43 \
    --run_time "run_20260531-135841"

python -m inference.non_pretrain_baseline.inference \
    --dataset_name "plantdoc" \
    --type_model "non_pretrain_baseline" \
    --model_name "mobilenetv3small_torchvision" \
    --seed 44 \
    --run_time "run_20260531-135859"

python -m inference.non_pretrain_baseline.inference \
    --dataset_name "plantdoc" \
    --type_model "non_pretrain_baseline" \
    --model_name "mobilenetv3small_torchvision" \
    --seed 45 \
    --run_time "run_20260531-135915"

python -m inference.non_pretrain_baseline.inference \
    --dataset_name "plantdoc" \
    --type_model "non_pretrain_baseline" \
    --model_name "mobilenetv3small_torchvision" \
    --seed 46 \
    --run_time "run_20260531-135935"


python -m inference.non_pretrain_baseline.inference \
    --dataset_name "plantdoc" \
    --type_model "non_pretrain_baseline" \
    --model_name "mobilenetv3small_timm" \
    --seed 42 \
    --run_time "run_20260531-155105"

python -m inference.non_pretrain_baseline.inference \
    --dataset_name "plantdoc" \
    --type_model "non_pretrain_baseline" \
    --model_name "mobilenetv3small_timm" \
    --seed 43 \
    --run_time "run_20260531-155531"

python -m inference.non_pretrain_baseline.inference \
    --dataset_name "plantdoc" \
    --type_model "non_pretrain_baseline" \
    --model_name "mobilenetv3small_timm" \
    --seed 44 \
    --run_time "run_20260531-155549"

python -m inference.non_pretrain_baseline.inference \
    --dataset_name "plantdoc" \
    --type_model "non_pretrain_baseline" \
    --model_name "mobilenetv3small_timm" \
    --seed 45 \
    --run_time "run_20260531-155620"

python -m inference.non_pretrain_baseline.inference \
    --dataset_name "plantdoc" \
    --type_model "non_pretrain_baseline" \
    --model_name "mobilenetv3small_timm" \
    --seed 46 \
    --run_time "run_20260531-155640"