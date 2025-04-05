"""
Dataset handling and preprocessing for BrainDiffusion.

This module implements data loading, preprocessing, and patch-based 
sampling for 3D brain MRI volumes using TorchIO.
"""

import os
import random
import torch
import torchio as tio
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def load_subjects(dataset_path, max_subjects=None):
    """Load NIfTI subjects from directory.
    
    Args:
        dataset_path: Path to directory containing .nii.gz files
        max_subjects: Maximum number of subjects to load (for testing)
        
    Returns:
        List of TorchIO Subject objects
    """
    subjects = []
    nifti_files = sorted([os.path.join(dataset_path, f) for f in os.listdir(dataset_path) 
                         if f.endswith('.nii.gz')])
    
    if max_subjects:
        nifti_files = nifti_files[:max_subjects]
    
    for file_path in tqdm(nifti_files, desc="Loading subjects"):
        subject = tio.Subject(
            mri=tio.ScalarImage(file_path),
            name=os.path.basename(file_path)
        )
        subjects.append(subject)
    
    print(f"Loaded {len(subjects)} subjects from {dataset_path}")
    return subjects


def prepare_transforms(augment=True):
    """Prepare preprocessing and augmentation transforms.
    
    Args:
        augment: Whether to apply data augmentation
        
    Returns:
        TorchIO transforms
    """
    # Preprocessing transforms applied to all data
    preprocess = tio.Compose([
        tio.RescaleIntensity(out_min_max=(0, 1)),
        tio.Resize((128, 128, 128)),  # Reduce size for memory efficiency
    ])
    
    # Augmentation transforms applied only to training data
    augmentation = tio.Compose([
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10),
        tio.RandomFlip(axes=(0,)),
        tio.RandomNoise(std=0.03),
        tio.RandomBiasField(coefficients=0.5),
    ])
    
    # Combine transforms
    if augment:
        return tio.Compose([preprocess, augmentation])
    else:
        return preprocess


def label_subjects_by_region(subjects, regions_dict=None):
    """Label subjects by brain region based on filename or processing.
    
    In a real scenario, you would analyze the image to determine region.
    For demonstration, we use filename patterns or random assignment.
    
    Args:
        subjects: List of TorchIO subjects
        regions_dict: Dictionary mapping region names to indices
        
    Returns:
        Labeled subjects
    """
    if regions_dict is None:
        regions_dict = {
            'frontal': 0,
            'parietal': 1,
            'temporal': 2,
            'occipital': 3,
            'cerebellum': 4,
        }
    
    labeled_subjects = []
    
    for subject in subjects:
        # Extract region from filename or randomly assign
        filename = subject.name.lower()
        
        if 'front' in filename:
            region = 'frontal'
        elif 'pari' in filename:
            region = 'parietal'
        elif 'temp' in filename:
            region = 'temporal'
        elif 'occip' in filename:
            region = 'occipital'
        elif 'cereb' in filename:
            region = 'cerebellum'
        else:
            # Default to random region for demo
            region = random.choice(list(regions_dict.keys()))
        
        # Add region label to subject
        subject.region = regions_dict[region]
        labeled_subjects.append(subject)
    
    return labeled_subjects


class BrainMRIDataset(tio.SubjectsDataset):
    """Dataset that provides both image and region label."""
    
    def __getitem__(self, index):
        """Get a subject and its label.
        
        Returns:
            Dictionary with MRI volume and region label
        """
        subject = self._subjects[index]
        if self.transform is not None:
            subject = self.transform(subject)
        
        # If subject has a region attribute, return it
        region = getattr(subject, 'region', None)
        
        return {
            'mri': subject.mri[tio.DATA],
            'region': region
        }


def create_dataloaders(
    dataset_path, 
    batch_size=2, 
    patch_size=64, 
    samples_per_volume=8, 
    max_queue_length=300,
    train_split=0.8,
    max_subjects=None
):
    """Create training and validation dataloaders with patch-based sampling.
    
    Args:
        dataset_path: Path to directory with .nii.gz files
        batch_size: Batch size
        patch_size: Size of patches for training (e.g., 64 for 64x64x64)
        samples_per_volume: Number of patches to extract from each volume
        max_queue_length: Maximum queue length for patch-based sampling
        train_split: Fraction of data to use for training
        max_subjects: Maximum number of subjects to load
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_regions: Number of unique regions
    """
    # Load subjects
    subjects = load_subjects(dataset_path, max_subjects)
    
    # Prepare transforms
    train_transform = prepare_transforms(augment=True)
    val_transform = prepare_transforms(augment=False)
    
    # Split into training and validation
    n_train = int(len(subjects) * train_split)
    train_subjects = subjects[:n_train]
    val_subjects = subjects[n_train:]
    
    print(f"Training subjects: {len(train_subjects)}")
    print(f"Validation subjects: {len(val_subjects)}")
    
    # Label subjects by region
    regions_dict = {
        'frontal': 0,
        'parietal': 1,
        'temporal': 2,
        'occipital': 3,
        'cerebellum': 4,
    }
    
    train_subjects = label_subjects_by_region(train_subjects, regions_dict)
    val_subjects = label_subjects_by_region(val_subjects, regions_dict)
    
    # Create datasets
    training_dataset = BrainMRIDataset(train_subjects, transform=train_transform)
    validation_dataset = BrainMRIDataset(val_subjects, transform=val_transform)
    
    # Create patch-based queue for training
    patches_training_set = tio.Queue(
        subjects_dataset=training_dataset,
        max_length=max_queue_length,
        samples_per_volume=samples_per_volume,
        sampler=tio.data.UniformSampler(patch_size),
        num_workers=4,
    )
    
    # Create dataloaders
    train_loader = DataLoader(patches_training_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=1)  # Use batch size 1 for full volumes
    
    return train_loader, val_loader, len(regions_dict)


if __name__ == "__main__":
    # Test dataset loading
    import argparse
    
    parser = argparse.ArgumentParser(description="Test dataset loading")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to IXI dataset")
    parser.add_argument("--max_subjects", type=int, default=5, help="Maximum subjects to load for testing")
    
    args = parser.parse_args()
    
    # Create dataloaders
    train_loader, val_loader, num_regions = create_dataloaders(
        args.data_dir,
        max_subjects=args.max_subjects
    )
    
    # Test loading a batch
    batch = next(iter(train_loader))
    print(f"Batch shape: {batch['mri'].shape}")
    print(f"Region labels: {batch['region']}")
    print(f"Number of regions: {num_regions}")
    
    print("Dataset test complete!")
