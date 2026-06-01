source venv/bin/activate
cd src
python -m benchmark.get_acc_mF1_non_pretrain_models \
    --dataset_name "plantdoc" \
    --type_model "non_pretrain_models" \
    --model_name "mobilenetv3small_timm" \
    --csv_store_dir "/media/data/minhht/clustering_moe/results/non_pretrain" \
    --csv_filename "mobilenetv3small_timm.csv" \
    --export_csv