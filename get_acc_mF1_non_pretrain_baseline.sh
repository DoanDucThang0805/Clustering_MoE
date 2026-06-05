source venv/bin/activate
cd src
python -m benchmark.get_acc_mF1_non_pretrain_baseline \
    --dataset_name "plantdoc" \
    --type_model "non_pretrain_baseline" \
    --model_name "mobilenetv3small_torchvision" \
    --csv_store_dir "/media/data/minhht/clustering_moe/mean_acc_mF1_results/non_pretrain_baseline" \
    --csv_filename "mobilenetv3small_torchvision.csv" \
    --export_csv