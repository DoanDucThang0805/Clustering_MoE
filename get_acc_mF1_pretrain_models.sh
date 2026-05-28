source venv/bin/activate
cd src
python -m benchmark.get_acc_mF1_pretrain_models \
    --dataset_name "plantdoc" \
    --type_model "pretrain_models" \
    --model_name "mobilenetv3small_timm" \
    --csv_store_dir "/media/data/minhht/clustering_moe/results/pretrain" \
    --csv_filename "mobilenetv3small_timm.csv" \
    --export_csv