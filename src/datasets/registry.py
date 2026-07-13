"""Dataset registry — chọn module dataset theo dataset_name.

Gom if/else chọn PlantDoc / PlantVillage vào 1 chỗ để các script training/inference
không phải import cứng. Import lazy (chỉ dựng dataset của dataset được chọn, tránh
load cả 2 tại import-time).

    from datasets.registry import get_train_val, get_test, get_moe_build
    train_ds, val_ds = get_train_val(args.dataset_name)
    build = get_moe_build(args.dataset_name); train, val, test = build(use_context=True)
"""
from __future__ import annotations

_CLASSIF = {
    "plantdoc": "datasets.plantdoc_dataset",
    "plantvillage": "datasets.plantvillage_dataset",
}
_MOE = {
    "plantdoc": "datasets.plantdoc_dataset_moe",
    "plantvillage": "datasets.plantvillage_dataset_moe",
}


def _classif_module(dataset_name: str):
    import importlib
    key = (dataset_name or "plantdoc").lower()
    if key not in _CLASSIF:
        raise ValueError(f"Unknown dataset_name '{dataset_name}'. "
                         f"Chọn một trong: {list(_CLASSIF)}")
    return importlib.import_module(_CLASSIF[key])


def get_train_val(dataset_name: str):
    """Trả (train_dataset, validation_dataset) cho dense/cluster training."""
    m = _classif_module(dataset_name)
    return m.train_dataset, m.validation_dataset


def get_test(dataset_name: str):
    """Trả test_dataset (dense/cluster inference)."""
    return _classif_module(dataset_name).test_dataset


def get_moe_build(dataset_name: str):
    """Trả hàm build_datasets(use_context) của dataset MoE tương ứng."""
    import importlib
    key = (dataset_name or "plantdoc").lower()
    if key not in _MOE:
        raise ValueError(f"Unknown dataset_name '{dataset_name}'. "
                         f"Chọn một trong: {list(_MOE)}")
    return importlib.import_module(_MOE[key]).build_datasets


def get_embedding_datasets(dataset_name: str):
    """Trả (train, validation, test) extract-embedding dataset cho backbone extractor."""
    m = _classif_module(dataset_name)
    return (m.extract_train_embedding_dataset,
            m.extract_validation_embedding_dataset,
            m.extract_test_embedding_dataset)
