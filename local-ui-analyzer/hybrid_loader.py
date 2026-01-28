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
    
    def __init__(
        self,
        silicon_root: str,
        ueyes_root: str,
        split: str = 'train',
        transform: Optional[A.Compose] = None
    ):
        self.silicon_root = Path(silicon_root)
        self.ueyes_root = Path(ueyes_root)
        self.split = split
        self.transform = transform or get_transforms(is_train=(split == 'train'))
        self.map_transform = get_map_transforms()
        
        # Collect samples from both datasets
        self.samples: List[Tuple[Path, Path, int]] = []  # (image_path, map_path, source_label)
        self.silicon_count = 0
        self.ueyes_count = 0
        
        # Load Silicon samples
        self._load_silicon_samples()
        
        # Load Ueyes samples  
        self._load_ueyes_samples()
        
        print(f"[HybridDataset] Loaded {self.silicon_count} Silicon + {self.ueyes_count} Ueyes = {len(self.samples)} total samples")
    
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
            
            if not images_dir.exists():
                warnings.warn(f"Silicon images directory not found: {images_dir}")
                continue
            
            if not maps_dir.exists():
                warnings.warn(f"Silicon maps directory not found: {maps_dir}")
                continue
            
            # Build map lookup (basename without extension -> full path)
            map_lookup = {}
            for map_file in maps_dir.iterdir():
                if map_file.is_file() and map_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    map_lookup[map_file.stem.lower()] = map_file
            
            # Match images to maps
            for img_file in images_dir.iterdir():
                if not img_file.is_file():
                    continue
                if img_file.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                    continue
                
                basename = img_file.stem.lower()
                if basename in map_lookup:
                    self.samples.append((img_file, map_lookup[basename], self.SILICON_LABEL))
                    self.silicon_count += 1
                else:
                    # Skip silently for efficiency (common in large datasets)
                    pass
    
    def _load_ueyes_samples(self):
        """Load samples from Ueyes dataset (flat structure but maps in subfolders)."""
        print("[HybridDataset] Scanning Ueyes dataset...")
        images_dir = self.ueyes_root / "images"
        
        # Maps are located in subfolders (heatmaps_7s is standard GT)
        # We prefer heatmaps_7s, but check others if missing
        potential_map_dirs = [
            self.ueyes_root / "saliency_maps" / "heatmaps_7s",
            self.ueyes_root / "saliency_maps" / "heatmaps_3s",
            self.ueyes_root / "saliency_maps" / "heatmaps_1s",
            self.ueyes_root / "saliency_maps"  # Fallback to root if user reorganized
        ]
        
        maps_dir = None
        for p_dir in potential_map_dirs:
            if p_dir.exists():
                maps_dir = p_dir
                print(f"[Ueyes] Using map directory: {maps_dir}")
                break
        
        if not images_dir.exists():
            warnings.warn(f"Ueyes images directory not found: {images_dir}")
            return
        
        if maps_dir is None:
            warnings.warn(f"Ueyes saliency_maps directory not found (checked heatmaps_7s/3s/1s/root)")
            return
        
        # Build map lookup (basename without extension -> full path)
        map_lookup = {}
        for map_file in maps_dir.iterdir():
            if map_file.is_file() and map_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                map_lookup[map_file.stem.lower()] = map_file
        
        # Match images to maps
        for img_file in images_dir.iterdir():
            if not img_file.is_file():
                continue
            if img_file.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                continue
            
            basename = img_file.stem.lower()
            if basename in map_lookup:
                self.samples.append((img_file, map_lookup[basename], self.UEYES_LABEL))
                self.ueyes_count += 1
            else:
                warnings.warn(f"[Ueyes] Missing map for: {img_file.name}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Get a sample by index.
        
        Returns:
            Tuple of (image_tensor, map_tensor, source_label)
            - image_tensor: (3, H, W) normalized RGB tensor
            - map_tensor: (1, H, W) grayscale saliency map [0, 1]
            - source_label: 0 for Silicon, 1 for Ueyes
        """
        img_path, map_path, source_label = self.samples[idx]
        
        try:
            # Load image as RGB
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            
            # Load map as grayscale
            saliency_map = Image.open(map_path).convert('L')
            saliency_map = np.array(saliency_map).astype(np.float32) / 255.0  # Normalize to [0, 1]
            
            # Apply transforms
            transformed = self.transform(image=image)
            image_tensor = transformed['image']
            
            map_transformed = self.map_transform(image=saliency_map)
            map_tensor = map_transformed['image']
            
            # Ensure map is (1, H, W)
            if map_tensor.dim() == 2:
                map_tensor = map_tensor.unsqueeze(0)
            
            return image_tensor, map_tensor, source_label
            
        except Exception as e:
            warnings.warn(f"Error loading sample {idx} ({img_path}): {e}")
            # Return a dummy sample to avoid crashing
            dummy_img = torch.zeros(3, TARGET_HEIGHT, TARGET_WIDTH)
            dummy_map = torch.zeros(1, TARGET_HEIGHT, TARGET_WIDTH)
            return dummy_img, dummy_map, source_label


def get_balanced_sampler(dataset: HybridDataset) -> WeightedRandomSampler:
    """
    Create a weighted random sampler for balanced 50/50 sampling between datasets.
    
    This is CRITICAL to prevent the larger Silicon dataset from drowning out 
    the smaller Ueyes (UI) dataset during training.
    
    Args:
        dataset: HybridDataset instance
    
    Returns:
        WeightedRandomSampler configured for balanced sampling
    """
    # Calculate weights for each source
    silicon_weight = 1.0 / max(dataset.silicon_count, 1)
    ueyes_weight = 1.0 / max(dataset.ueyes_count, 1)
    
    print(f"[BalancedSampler] Silicon weight: {silicon_weight:.6f} ({dataset.silicon_count} samples)")
    print(f"[BalancedSampler] Ueyes weight: {ueyes_weight:.6f} ({dataset.ueyes_count} samples)")
    
    # Assign weight to each sample based on its source
    weights = []
    for _, _, source_label in dataset.samples:
        if source_label == HybridDataset.SILICON_LABEL:
            weights.append(silicon_weight)
        else:
            weights.append(ueyes_weight)
    
    weights = torch.tensor(weights, dtype=torch.float64)
    
    # Number of samples to draw per epoch
    # Use 2x the smaller dataset size to ensure good coverage
    num_samples = 2 * min(dataset.silicon_count, dataset.ueyes_count)
    num_samples = max(num_samples, len(dataset))  # At least full dataset size
    
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=num_samples,
        replacement=True  # Required for weighted sampling
    )
    
    print(f"[BalancedSampler] Will draw {num_samples} samples per epoch")
    
    return sampler


# Quick test
if __name__ == "__main__":
    # Test loading from default paths
    silicon_path = "models/Datasets/Silicon"
    ueyes_path = "models/Datasets/Ueyes"
    
    print("Testing HybridDataset...")
    dataset = HybridDataset(silicon_path, ueyes_path, split='train')
    
    if len(dataset) > 0:
        img, smap, label = dataset[0]
        print(f"Sample 0: image shape={img.shape}, map shape={smap.shape}, label={label}")
        
        print("\nTesting balanced sampler...")
        sampler = get_balanced_sampler(dataset)
        print("Sampler created successfully!")
    else:
        print("WARNING: No samples loaded!")
