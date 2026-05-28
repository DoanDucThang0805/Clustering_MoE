import os
import hashlib
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Literal
import cv2
import numpy as np
from PIL import Image

from .plantdoc_dataset import train_dataset, validation_dataset, test_dataset


class DataLeakChecker:
    """
    Comprehensive data leak detection for train/validation/test splits.
    
    Detects:
    - Image path leakage (same file in multiple splits)
    - Exact duplicates (byte-identical files)
    - Perceptual duplicates (visually similar images)
    - Filename conflicts
    
    Parameters:
    -----------
    datasets : Dict[str, Dataset], optional
        Dictionary with keys ('train', 'validation', 'test') and dataset objects.
        If None, uses default plantdoc datasets.
    
    perceptual_threshold : float, default=0.95
        Similarity threshold for perceptual hash matching (0-1).
        Higher = stricter matching.
    
    verbose : bool, default=True
        Enable detailed logging to console.
    
    log_level : str, default='INFO'
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
    
    hash_algorithm : str, default='md5'
        File hash algorithm ('md5', 'sha256').
    
    phash_method : str, default='phash'
        Perceptual hash method ('phash', 'dhash').
    
    Examples:
    ---------
    >>> checker = DataLeakChecker()
    >>> results = checker.run_all_checks()
    
    >>> checker = DataLeakChecker(
    ...     perceptual_threshold=0.90,
    ...     log_level='DEBUG'
    ... )
    >>> path_leaks = checker.check_image_path_leakage()
    """
    
    def __init__(
        self,
        datasets: Optional[Dict[str, object]] = None,
        perceptual_threshold: float = 0.95,
        verbose: bool = True,
        log_level: str = 'INFO',
        hash_algorithm: str = 'md5',
        phash_method: str = 'phash'
    ):
        """Initialize DataLeakChecker with parameters."""
        self.perceptual_threshold = perceptual_threshold
        self.hash_algorithm = hash_algorithm
        self.phash_method = phash_method
        self.verbose = verbose
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        if verbose and not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        # Setup datasets
        if datasets is None:
            self.datasets = {
                'train': train_dataset,
                'validation': validation_dataset,
                'test': test_dataset
            }
        else:
            self.datasets = datasets
        
        self.results = {}
    
    def _compute_file_hash(self, file_path: str) -> Optional[str]:
        """Compute file hash."""
        try:
            hash_obj = hashlib.new(self.hash_algorithm)
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            self.logger.error(f"Error hashing {file_path}: {e}")
            return None
    
    def _compute_image_hash(self, image_path: str) -> Optional[str]:
        """Compute perceptual hash of image."""
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            
            img = cv2.resize(img, (8, 8))
            
            if self.phash_method == 'phash':
                mean = img.mean()
                phash = ''.join(['1' if pixel > mean else '0' 
                               for pixel in img.flatten()])
                return phash
            elif self.phash_method == 'dhash':
                dhash = ''
                for row in img:
                    for i in range(len(row) - 1):
                        dhash += '1' if row[i] > row[i + 1] else '0'
                return dhash
        except Exception as e:
            self.logger.error(f"Error computing image hash for {image_path}: {e}")
            return None
    
    def _hamming_distance(self, str1: str, str2: str) -> int:
        """Compute Hamming distance between two strings."""
        return sum(c1 != c2 for c1, c2 in zip(str1, str2))
    
    def _log_section(self, title: str):
        """Log a section header."""
        self.logger.info("\n" + "="*80)
        self.logger.info(title)
        self.logger.info("="*80)
    
    def _log_result(self, message: str, level: str = 'info'):
        """Log result message."""
        log_func = getattr(self.logger, level.lower(), self.logger.info)
        log_func(message)
    
    def generate_statistics(self) -> Dict:
        """Generate comprehensive statistics for all datasets."""
        self._log_section("DATASET STATISTICS")
        
        stats = {}
        
        for split_name, dataset in self.datasets.items():
            self._log_result(f"\n{split_name.upper()} Set:")
            self._log_result(f"  Total images: {len(dataset)}")
            self._log_result(f"  Classes: {len(dataset.class_to_idx)}")
            
            class_counts = defaultdict(int)
            for label in dataset.labels:
                class_name = dataset.idx_to_class[label]
                class_counts[class_name] += 1
            
            self._log_result(f"  Class distribution:")
            for class_name in sorted(class_counts.keys()):
                count = class_counts[class_name]
                percentage = (count / len(dataset)) * 100
                self._log_result(f"    - {class_name}: {count} ({percentage:.1f}%)")
            
            stats[split_name] = {
                'total': len(dataset),
                'classes': len(dataset.class_to_idx),
                'class_distribution': dict(class_counts)
            }
        
        return stats
    
    def check_image_path_leakage(self) -> Dict:
        """Check for identical image paths across datasets."""
        self._log_section("CHECKING IMAGE PATH LEAKAGE (CRITICAL)")
        
        all_paths = {}
        path_leaks = []
        
        for split_name, dataset in self.datasets.items():
            self._log_result(f"\nScanning {split_name} set ({len(dataset)} images)...")
            for idx, img_path in enumerate(dataset.image_paths):
                if (idx + 1) % 100 == 0 or (idx + 1) == len(dataset):
                    self.logger.debug(f"  Progress: {idx + 1}/{len(dataset)}")
                
                normalized_path = os.path.normpath(img_path)
                
                if normalized_path in all_paths:
                    prev_info = all_paths[normalized_path]
                    path_leaks.append({
                        'path': normalized_path,
                        'split1': prev_info['split'],
                        'split2': split_name,
                        'class1': prev_info['class'],
                        'class2': dataset.idx_to_class[dataset.labels[idx]],
                    })
                else:
                    all_paths[normalized_path] = {
                        'split': split_name,
                        'label': dataset.labels[idx],
                        'class': dataset.idx_to_class[dataset.labels[idx]]
                    }
        
        self._log_result(f"\nTotal unique image paths: {len(all_paths)}")
        self._log_result(f"Image path leaks: {len(path_leaks)}")
        
        if path_leaks:
            self._log_result(
                f"\n❌ CRITICAL: Found {len(path_leaks)} IMAGE PATH LEAKS!\n",
                level='error'
            )
            for idx, leak in enumerate(path_leaks, 1):
                self._log_result(f"Path Leak #{idx}:", level='error')
                self._log_result(f"  File: {leak['path']}", level='error')
                self._log_result(
                    f"  In {leak['split1']} ({leak['class1']}) AND {leak['split2']} ({leak['class2']})",
                    level='error'
                )
        else:
            self._log_result("✓ No image path leakage detected!")
        
        result = {
            'total_paths': len(all_paths),
            'leaks_found': len(path_leaks),
            'leaks': path_leaks
        }
        self.results['path_leaks'] = result
        return result
    
    def check_exact_duplicates(self) -> Dict:
        """Check for exact duplicate files (byte-identical)."""
        self._log_section(f"CHECKING EXACT DUPLICATES ({self.hash_algorithm.upper()} hash)")
        
        hash_to_images = defaultdict(list)
        duplicates_found = []
        
        for split_name, dataset in self.datasets.items():
            self._log_result(f"\nComputing {self.hash_algorithm} hashes for {split_name} set...")
            for idx, img_path in enumerate(dataset.image_paths):
                if (idx + 1) % 100 == 0 or (idx + 1) == len(dataset):
                    self.logger.debug(f"  Progress: {idx + 1}/{len(dataset)}")
                
                file_hash = self._compute_file_hash(img_path)
                if file_hash:
                    hash_to_images[file_hash].append({
                        'split': split_name,
                        'path': img_path,
                        'label': dataset.labels[idx],
                        'class': dataset.idx_to_class[dataset.labels[idx]]
                    })
        
        for file_hash, images in hash_to_images.items():
            if len(images) > 1:
                splits = set([img['split'] for img in images])
                if len(splits) > 1:
                    duplicates_found.append(images)
        
        self._log_result(f"\nTotal unique files: {len(hash_to_images)}")
        self._log_result(f"Exact duplicate groups: {len(duplicates_found)}")
        
        if duplicates_found:
            self._log_result(
                f"\n⚠️  ALERT: Found {len(duplicates_found)} exact duplicate groups!\n",
                level='warning'
            )
            for idx, images in enumerate(duplicates_found, 1):
                self._log_result(f"Duplicate Group #{idx}:", level='warning')
                for img in images:
                    self._log_result(
                        f"  - [{img['split']}] {Path(img['path']).name} (class: {img['class']})",
                        level='warning'
                    )
                self._log_result("", level='warning')
        else:
            self._log_result("✓ No exact duplicates found!")
        
        result = {
            'total_unique': len(hash_to_images),
            'duplicates_found': len(duplicates_found),
            'duplicates': duplicates_found
        }
        self.results['exact_duplicates'] = result
        return result
    
    def check_perceptual_duplicates(self) -> Dict:
        """Check for perceptually similar images."""
        self._log_section(
            f"CHECKING PERCEPTUAL DUPLICATES ({self.phash_method}, "
            f"threshold={self.perceptual_threshold})"
        )
        
        phash_to_images = defaultdict(list)
        similar_found = []
        
        for split_name, dataset in self.datasets.items():
            self._log_result(f"\nComputing perceptual hashes for {split_name} set...")
            for idx, img_path in enumerate(dataset.image_paths):
                if (idx + 1) % 100 == 0 or (idx + 1) == len(dataset):
                    self.logger.debug(f"  Progress: {idx + 1}/{len(dataset)}")
                
                phash = self._compute_image_hash(img_path)
                if phash:
                    phash_to_images[phash].append({
                        'split': split_name,
                        'path': img_path,
                        'label': dataset.labels[idx],
                        'class': dataset.idx_to_class[dataset.labels[idx]],
                        'phash': phash
                    })
        
        hashes_list = list(phash_to_images.keys())
        for i, hash1 in enumerate(hashes_list):
            for hash2 in hashes_list[i+1:]:
                distance = self._hamming_distance(hash1, hash2)
                similarity = 1 - (distance / len(hash1))
                
                if similarity >= self.perceptual_threshold:
                    images1 = phash_to_images[hash1]
                    images2 = phash_to_images[hash2]
                    
                    splits1 = set([img['split'] for img in images1])
                    splits2 = set([img['split'] for img in images2])
                    
                    if len(splits1 & splits2) == 0 or len(splits1.union(splits2)) > 1:
                        similar_found.append({
                            'similarity': similarity,
                            'images1': images1,
                            'images2': images2
                        })
        
        self._log_result(f"\nTotal unique perceptual hashes: {len(phash_to_images)}")
        self._log_result(f"Similar image pairs: {len(similar_found)}")
        
        if similar_found:
            self._log_result(
                f"\n⚠️  ALERT: Found {len(similar_found)} similar image pairs!\n",
                level='warning'
            )
            for idx, pair in enumerate(similar_found, 1):
                self._log_result(
                    f"Similar Pair #{idx} (similarity: {pair['similarity']:.4f}):",
                    level='warning'
                )
                for img in pair['images1']:
                    self._log_result(
                        f"  - [{img['split']}] {Path(img['path']).name} (class: {img['class']})",
                        level='warning'
                    )
                for img in pair['images2']:
                    self._log_result(
                        f"  - [{img['split']}] {Path(img['path']).name} (class: {img['class']})",
                        level='warning'
                    )
                self._log_result("", level='warning')
        else:
            self._log_result("✓ No similar images found!")
        
        result = {
            'total_unique_hashes': len(phash_to_images),
            'similar_pairs': len(similar_found),
            'similar_images': similar_found
        }
        self.results['perceptual_duplicates'] = result
        return result
    
    def check_filename_similarity(self) -> Dict:
        """Check for identical filenames across datasets."""
        self._log_section("CHECKING FILENAME SIMILARITY")
        
        filename_to_info = defaultdict(list)
        identical_filenames = []
        
        for split_name, dataset in self.datasets.items():
            for idx, img_path in enumerate(dataset.image_paths):
                filename = os.path.basename(img_path)
                filename_to_info[filename].append({
                    'split': split_name,
                    'path': img_path,
                    'label': dataset.labels[idx],
                    'class': dataset.idx_to_class[dataset.labels[idx]]
                })
        
        for filename, images in filename_to_info.items():
            if len(images) > 1:
                splits = set([img['split'] for img in images])
                if len(splits) > 1:
                    identical_filenames.append((filename, images))
        
        self._log_result(f"\nTotal unique filenames: {len(filename_to_info)}")
        self._log_result(f"Identical filenames: {len(identical_filenames)}")
        
        if identical_filenames:
            self._log_result(
                f"\n⚠️  ALERT: Found {len(identical_filenames)} identical filenames!\n",
                level='warning'
            )
            for filename, images in identical_filenames:
                self._log_result(f"File: {filename}", level='warning')
                for img in images:
                    self._log_result(
                        f"  - [{img['split']}] {img['path']} (class: {img['class']})",
                        level='warning'
                    )
                self._log_result("", level='warning')
        else:
            self._log_result("✓ No identical filenames found!")
        
        result = {
            'total_unique_filenames': len(filename_to_info),
            'identical_filenames': len(identical_filenames),
            'filename_conflicts': identical_filenames
        }
        self.results['filename_duplicates'] = result
        return result
    
    def run_all_checks(self) -> Dict:
        """Run all data leak checks."""
        self._log_section("STARTING COMPREHENSIVE DATA LEAK CHECK")
        
        stats = self.generate_statistics()
        path_leaks = self.check_image_path_leakage()
        exact_dups = self.check_exact_duplicates()
        perceptual_dups = self.check_perceptual_duplicates()
        filename_dups = self.check_filename_similarity()
        
        # Summary
        self._log_section("FINAL SUMMARY")
        self._log_result(f"\n✓ Image path leaks: {path_leaks['leaks_found']}")
        self._log_result(f"✓ Exact duplicates: {exact_dups['duplicates_found']}")
        self._log_result(f"✓ Perceptual duplicates: {perceptual_dups['similar_pairs']}")
        self._log_result(f"✓ Filename conflicts: {filename_dups['identical_filenames']}")
        
        total_leaks = (path_leaks['leaks_found'] + exact_dups['duplicates_found'] + 
                      perceptual_dups['similar_pairs'] + filename_dups['identical_filenames'])
        
        if total_leaks == 0:
            self._log_result("\n✅ CONCLUSION: NO DATA LEAKAGE DETECTED!\n")
            self._log_result("Your train/validation/test splits are properly separated.")
        else:
            self._log_result("\n❌ CONCLUSION: DATA LEAKAGE DETECTED!\n", level='warning')
            if path_leaks['leaks_found'] > 0:
                self._log_result(
                    f"  - {path_leaks['leaks_found']} IMAGE PATH LEAKS (CRITICAL!)",
                    level='error'
                )
            if exact_dups['duplicates_found'] > 0:
                self._log_result(
                    f"  - {exact_dups['duplicates_found']} exact duplicate groups",
                    level='warning'
                )
            if perceptual_dups['similar_pairs'] > 0:
                self._log_result(
                    f"  - {perceptual_dups['similar_pairs']} similar image pairs",
                    level='warning'
                )
            if filename_dups['identical_filenames'] > 0:
                self._log_result(
                    f"  - {filename_dups['identical_filenames']} identical filenames",
                    level='warning'
                )
            self._log_result(
                "This can lead to overly optimistic performance estimates.",
                level='warning'
            )
        
        return {
            'statistics': stats,
            'path_leaks': path_leaks,
            'exact_duplicates': exact_dups,
            'perceptual_duplicates': perceptual_dups,
            'filename_duplicates': filename_dups,
            'total_leaks': total_leaks
        }


# ============================================================================
# COMMAND LINE USAGE
# ============================================================================

if __name__ == '__main__':
    import sys
    
    # Parse arguments
    perceptual_threshold = 0.95
    log_level = 'INFO'
    hash_algorithm = 'md5'
    phash_method = 'phash'
    
    for arg in sys.argv[1:]:
        if arg.startswith('--threshold='):
            perceptual_threshold = float(arg.split('=')[1])
        elif arg.startswith('--log-level='):
            log_level = arg.split('=')[1].upper()
        elif arg.startswith('--hash='):
            hash_algorithm = arg.split('=')[1]
        elif arg.startswith('--phash-method='):
            phash_method = arg.split('=')[1]
    
    # Run checks
    checker = DataLeakChecker(
        perceptual_threshold=perceptual_threshold,
        log_level=log_level,
        hash_algorithm=hash_algorithm,
        phash_method=phash_method
    )
    results = checker.run_all_checks()
