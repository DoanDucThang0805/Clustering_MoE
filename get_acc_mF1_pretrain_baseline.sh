source venv/bin/activate
cd src
python -m benchmark.get_acc_mF1_pretrain_baseline \
    --dataset_name "plantdoc" \
    --type_model "pretrain_baseline" \
    --model_name "mobilenetv3small_torchvision" \
    --csv_store_dir "/media/data/minhht/clustering_moe/mean_acc_mF1_results/pretrain_baseline" \
    --csv_filename "mobilenetv3small_torchvision.csv" \
    --export_csv