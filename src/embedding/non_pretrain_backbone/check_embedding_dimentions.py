import os
import numpy as np
from pathlib import Path

# Path to the embeddings directory
embeddings_dir = "/media/data/minhht/clustering_moe/feature_embeddings/plantdoc/non_pretrain_models/mobilenetv3small_torchvision"

# Get all seed directories
seed_dirs = sorted([d for d in os.listdir(embeddings_dir) if d.startswith('seed_')])

print("=" * 80)
print("EMBEDDING DIMENSIONS FOR EACH SEED")
print("=" * 80)

for seed_dir in seed_dirs:
    seed_path = os.path.join(embeddings_dir, seed_dir)
    print(f"\n{seed_dir}:")
    print("-" * 40)
    
    # Get all .npz files in the seed directory
    npz_files = sorted([f for f in os.listdir(seed_path) if f.endswith('.npz')])
    
    for npz_file in npz_files:
        npz_path = os.path.join(seed_path, npz_file)
        
        # Load the npz file
        data = np.load(npz_path)
        
        # Get all arrays in the npz file
        for key in data.files:
            array = data[key]
            print(f"  {npz_file} -> {key}: {array.shape}")

print("\n" + "=" * 80)



