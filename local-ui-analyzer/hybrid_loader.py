"""
Hybrid Dataset Loader for EML-NET Training
Combines Silicon (natural images) and Ueyes (UI screenshots) datasets 
with weighted sampling for balanced training.

Author: Generated for EML-NET Hybrid Training Pipeline
"""

import os
import warnings
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Target resolution (W, H) = (640, 480)
TARGET_WIDTH = 640
TARGET_HEIGHT = 480


def get_transforms(is_train: bool = True) -> A.Compose:
    """
    Get albumentations transforms for image preprocessing.
    
    Args:
        is_train: If True, applies training augmentations (not used in base config,
                  but can be extended with flips, color jitter, etc.)
    
    Returns:
        Albumentations Compose pipeline
    """
    transforms = [
        A.Resize(height=TARGET_HEIGHT, width=TARGET_WIDTH),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ]
    
    return A.Compose(transforms)


def get_map_transforms() -> A.Compose:
    """
    Get transforms for saliency maps (grayscale, normalized to 0-1).
    
    Returns:
        Albumentations Compose pipeline for maps
    """
    return A.Compose([
        A.Resize(height=TARGET_HEIGHT, width=TARGET_WIDTH),
        ToTensorV2(),
    ])


class HybridDataset(Dataset):
    """
    Combined dataset for Silicon (natural) and Ueyes (UI) images.
    
    Handles different directory structures:
    - Silicon: Nested structure with train/val subdirectories
    - Ueyes: Flat structure with images/ and saliency_maps/
    
    Args:
        silicon_root: Path to Silicon dataset root (contains Images/ and maps/)
        ueyes_root: Path to Ueyes dataset root (contains images/ and saliency_maps/)
        split: Dataset split ('train', 'val', or 'all')
        transform: Optional custom transforms for images
    """
    
    # Source labels for tracking dataset origin
    SILICON_LABEL = 0
    UEYES_LABEL = 1
    MASSVIS_LABEL = 2
    FIWI_LABEL = 3
    
    def __init__(
        self,
        silicon_root: str,
        ueyes_root: str,
        massvis_root: Optional[str] = None,
        fiwi_root: Optional[str] = None,
        salchart_root: Optional[str] = None,
        split: str = 'train',
        transform: Optional[A.Compose] = None
    ):
        self.silicon_root = Path(silicon_root) if silicon_root else None
        self.ueyes_root = Path(ueyes_root) if ueyes_root else None
        self.massvis_root = Path(massvis_root) if massvis_root else None
        self.fiwi_root = Path(fiwi_root) if fiwi_root else None
        self.salchart_root = Path(salchart_root) if salchart_root else None
        
        self.split = split
        self.transform = transform or get_transforms(is_train=(split == 'train'))
        self.map_transform = get_map_transforms()
        
        # Collect samples from all datasets
        self.samples: List[Tuple[Path, Path, int]] = []  # (image_path, map_path, source_label)
        self.silicon_count = 0
        self.ueyes_count = 0
        self.massvis_count = 0
        self.fiwi_count = 0
        self.salchart_count = 0
        
        # Load samples
        if self.silicon_root: self._load_silicon_samples()
        if self.ueyes_root: self._load_ueyes_samples()
        if self.massvis_root: self._load_massvis_samples()
        if self.fiwi_root: self._load_mobile_ui_samples()
        if self.salchart_root: self._load_salchart_samples()
        
        print(f"[HybridDataset] Loaded: Silicon={self.silicon_count}, Ueyes={self.ueyes_count}, "
              f"MassVis={self.massvis_count}, FiWI={self.fiwi_count}, SalChart={self.salchart_count}. Total={len(self.samples)}")

    def _load_silicon_samples(self):
        """Load samples from Silicon dataset (nested train/val structure)."""
        print("[HybridDataset] Scanning Silicon dataset...")
        images_root = self.silicon_root / "Images"
        maps_root = self.silicon_root / "maps"
        
        # Determine which subdirectories to scan
        if self.split == 'all':
            subdirs = ['train', 'val']
        elif self.split in ['train', 'val']:
            subdirs = [self.split]
        else:
            warnings.warn(f"Unknown split '{self.split}' for Silicon, using 'train'")
            subdirs = ['train']
        
        for subdir in subdirs:
            images_dir = images_root / subdir
            maps_dir = maps_root / subdir
            
            if not images_dir.exists() or not maps_dir.exists():
                continue
            
            # Build map lookup
            map_lookup = {}
            for map_file in maps_dir.iterdir():
                if map_file.is_file() and map_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    map_lookup[map_file.stem.lower()] = map_file
            
            # Match images to maps
            for img_file in images_dir.iterdir():
                if not img_file.is_file() or img_file.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                    continue
                
                basename = img_file.stem.lower()
                if basename in map_lookup:
                    self.samples.append((img_file, map_lookup[basename], self.SILICON_LABEL))
                    self.silicon_count += 1

    def _load_ueyes_samples(self):
        """Load samples from Ueyes dataset."""
        print("[HybridDataset] Scanning Ueyes dataset...")
        images_dir = self.ueyes_root / "images"
        
        # Maps are located in subfolders (heatmaps_7s is standard GT)
        potential_map_dirs = [
            self.ueyes_root / "saliency_maps" / "heatmaps_7s",
            self.ueyes_root / "saliency_maps" / "heatmaps_3s",
            self.ueyes_root / "saliency_maps"
        ]
        
        maps_dir = None
        for p_dir in potential_map_dirs:
            if p_dir.exists():
                maps_dir = p_dir
                break
        
        if not images_dir.exists() or maps_dir is None:
            return
        
        # Build map lookup
        map_lookup = {}
        for map_file in maps_dir.iterdir():
            if map_file.is_file() and map_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                map_lookup[map_file.stem.lower()] = map_file
        
        # Match images to maps
        for img_file in images_dir.iterdir():
            if not img_file.is_file() or img_file.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                continue
            
            basename = img_file.stem.lower()
            if basename in map_lookup:
                self.samples.append((img_file, map_lookup[basename], self.UEYES_LABEL))
                self.ueyes_count += 1

    def _load_massvis_samples(self):
        """Load MassVis samples (Stub - Saliency maps currently missing)."""
        print("[HybridDataset] Scanning MassVis dataset...")
        if not self.massvis_root:
            return
            
        # Placeholder for when maps are available
        # Structure identified: vis1/, vis2/, vis3/ contain stimulus.
        # Targets/ contain stimulus.
        pass

    def _load_salchart_samples(self):
        """Load SALchart QA samples (Stub - Stimulus images currently missing)."""
        print("[HybridDataset] Scanning SALchart QA dataset...")
        # Placeholder for when stimulus images are available
        pass

    def _load_mobile_ui_samples(self):
        """Load Mobile UI Saliency samples."""
        print("[HybridDataset] Scanning Mobile UI Saliency dataset...")
        if not self.fiwi_root:
             return

        images_dir = self.fiwi_root / "ui_screenshots"
        maps_dir = self.fiwi_root / "density_maps"
        
        if not images_dir.exists() or not maps_dir.exists():
            warnings.warn(f"Mobile UI folders not found: {images_dir} or {maps_dir}")
            return
            
        # Map lookup: filenames usually match
        map_lookup = {p.stem.lower(): p for p in maps_dir.glob("*") if p.suffix.lower() in ['.png','.jpg','.jpeg']}
        
        count = 0
        for img_file in images_dir.glob("*"):
             if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                if img_file.stem.lower() in map_lookup:
                    self.samples.append((img_file, map_lookup[img_file.stem.lower()], self.FIWI_LABEL))
                    count += 1
        
        self.fiwi_count = count
        print(f"[Mobile UI] Loaded {count} samples")

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        img_path, map_path, source_label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            
            saliency_map = Image.open(map_path).convert('L')
            saliency_map = np.array(saliency_map).astype(np.float32) / 255.0
            
            transformed = self.transform(image=image)
            image_tensor = transformed['image']
            
            map_transformed = self.map_transform(image=saliency_map)
            map_tensor = map_transformed['image']
            
            if map_tensor.dim() == 2:
                map_tensor = map_tensor.unsqueeze(0)
            
            return image_tensor, map_tensor, source_label
        except Exception as e:
            warnings.warn(f"Error loading sample {idx}: {e}")
            return torch.zeros(3, TARGET_HEIGHT, TARGET_WIDTH), torch.zeros(1, TARGET_HEIGHT, TARGET_WIDTH), source_label


def get_balanced_sampler(dataset: HybridDataset) -> WeightedRandomSampler:
    """Create weighted sampler for N datasets."""
    counts = {
        HybridDataset.SILICON_LABEL: dataset.silicon_count,
        HybridDataset.UEYES_LABEL: dataset.ueyes_count,
        HybridDataset.MASSVIS_LABEL: dataset.massvis_count,
        HybridDataset.FIWI_LABEL: dataset.fiwi_count
    }
    
    # Calculate weight for each class: 1.0 / count
    weights_map = {}
    valid_datasets = 0
    for label, count in counts.items():
        if count > 0:
            weights_map[label] = 1.0 / count
            valid_datasets += 1
        else:
            weights_map[label] = 0.0
            
    if valid_datasets == 0:
        return None

    # Assign weights
    sample_weights = []
    for _, _, label in dataset.samples:
        sample_weights.append(weights_map[label])
    
    sample_weights = torch.tensor(sample_weights, dtype=torch.float64)
    
    # Total samples = k * min_dataset_size * num_datasets (heuristic)
    # Or simply: len(dataset) but balanced. 
    # Let's ensure we see every image from the smallest resource at least once per epoch essentially.
    active_counts = [c for c in counts.values() if c > 0]
    num_samples = max(len(dataset), len(active_counts) * min(active_counts) * 2)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True
    )
    return sampler

if __name__ == "__main__":
    # Test paths (relative to project root)
    dataset = HybridDataset(
        silicon_root="models/Datasets/Silicon",
        ueyes_root="models/Datasets/Ueyes",
        massvis_root="models/Datasets/massvis",
        fiwi_root="models/Datasets/mobile ui salency",
        salchart_root="models/Datasets/SALchart QA"
    )
    print(f"Total loaded: {len(dataset)}")
