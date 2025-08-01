# MedSAM Kidney Segmentation with KiTS23 Dataset

## Project Overview
This project implements kidney tumor segmentation using MedSAM (Medical SAM) fine-tuned on the KiTS23 (Kidney Tumor Segmentation Challenge 2023) dataset. The pipeline converts 3D medical images to 2D slices, processes them through MedSAM, and applies post-processing to reconstruct 3D segmentations.

## Dataset
- **KiTS23 Dataset**: Kidney Tumor Segmentation Challenge 2023
- **Format**: 3D CT scans with corresponding segmentation masks
- **Location**: `kits23/` directory in this repository

## Complete Pipeline

### 1. Data Preprocessing
#### 3D to 2D Conversion
- Extract 2D slices from 3D CT volumes along axial, sagittal, and coronal planes
- Normalize Hounsfield Units (HU) to appropriate ranges for MedSAM
- Apply windowing (e.g., abdominal window: -160 to 240 HU)
- Resize slices to MedSAM input dimensions (1024x1024)

#### Data Organization
```
data/
├── raw_3d/           # Original KiTS23 3D volumes
├── processed_2d/     # Converted 2D slices
│   ├── images/       # CT slices
│   ├── masks/        # Corresponding segmentation masks
│   └── metadata/     # Slice information and coordinates
└── splits/           # Train/val/test splits
```

### 2. MedSAM Integration
#### Model Setup
- Load pre-trained MedSAM model
- Configure for kidney segmentation task
- Set up prompt engineering for kidney region detection

#### Fine-tuning Strategy
- **Prompt Types**: Point prompts, bounding box prompts
- **Loss Function**: Dice loss + Cross-entropy loss
- **Optimizer**: AdamW with learning rate scheduling
- **Data Augmentation**: Rotation, scaling, intensity variations

### 3. Training Pipeline
#### Phase 1: Pre-training
- Train on general medical image segmentation tasks
- Use diverse medical datasets for robust feature learning

#### Phase 2: Fine-tuning
- Fine-tune specifically on KiTS23 kidney data
- Implement curriculum learning (easy to hard cases)
- Use mixed precision training for efficiency

#### Phase 3: Specialized Training
- Focus on challenging cases (small tumors, complex boundaries)
- Implement active learning for difficult samples

### 4. Inference Pipeline
#### 2D Processing
- Process each 2D slice through MedSAM
- Apply prompt-based segmentation
- Generate confidence scores for each prediction

#### 3D Reconstruction
- Reconstruct 3D volumes from 2D predictions
- Apply consistency checks across adjacent slices
- Implement 3D morphological operations

### 5. Post-processing
#### 2D Post-processing
- Remove small false positive regions
- Fill holes in segmentation masks
- Apply morphological operations (erosion, dilation)

#### 3D Post-processing
- **Volume Consistency**: Ensure anatomical consistency across slices
- **Spatial Smoothing**: Apply 3D Gaussian filtering
- **Connected Components**: Remove isolated small regions
- **Anatomical Constraints**: Apply kidney-specific anatomical rules

#### Quality Assurance
- Validate segmentation boundaries
- Check for anatomical plausibility
- Implement confidence-based filtering

### 6. Evaluation Metrics
#### 2D Metrics
- Dice Coefficient
- IoU (Intersection over Union)
- Hausdorff Distance
- Average Surface Distance

#### 3D Metrics
- 3D Dice Coefficient
- Volume Similarity
- Surface Distance Metrics
- Tumor-specific metrics (if applicable)

## File Structure
```
MedSAM_Kidney_Segmentation/
├── README.md
├── kits23/                    # KiTS23 dataset
├── src/
│   ├── data/
│   │   ├── preprocessing.py   # 3D to 2D conversion
│   │   ├── augmentation.py    # Data augmentation
│   │   └── dataloader.py     # Custom data loaders
│   ├── models/
│   │   ├── medsam.py         # MedSAM model wrapper
│   │   └── fine_tuning.py    # Fine-tuning logic
│   ├── training/
│   │   ├── trainer.py         # Training loop
│   │   └── losses.py         # Loss functions
│   ├── inference/
│   │   ├── inference.py      # 2D inference
│   │   └── reconstruction.py # 3D reconstruction
│   └── postprocessing/
│       ├── postprocess_2d.py # 2D post-processing
│       └── postprocess_3d.py # 3D post-processing
├── configs/
│   ├── data_config.yaml      # Data configuration
│   ├── model_config.yaml     # Model configuration
│   └── training_config.yaml  # Training parameters
├── scripts/
│   ├── preprocess_data.py    # Data preprocessing script
│   ├── train.py             # Training script
│   └── evaluate.py          # Evaluation script
└── results/
    ├── checkpoints/         # Model checkpoints
    ├── predictions/         # Model predictions
    └── logs/               # Training logs
```

## Implementation Steps

### Step 1: Environment Setup
```bash
# Install dependencies
pip install torch torchvision
pip install monai
pip install medsam
pip install nibabel
pip install scikit-image
```

### Step 2: Data Preprocessing
```bash
python scripts/preprocess_data.py --input kits23/ --output data/processed_2d/
```

### Step 3: Training
```bash
python scripts/train.py --config configs/training_config.yaml
```

### Step 4: Inference
```bash
python scripts/evaluate.py --model_path results/checkpoints/best_model.pth --data_path data/processed_2d/
```

## Key Challenges and Solutions

### Challenge 1: 3D to 2D Conversion
- **Issue**: Loss of spatial context when converting 3D to 2D
- **Solution**: Include slice position information and adjacent slice context

### Challenge 2: MedSAM Prompt Engineering
- **Issue**: Optimal prompt selection for kidney segmentation
- **Solution**: Implement automated prompt generation based on image features

### Challenge 3: 3D Reconstruction
- **Issue**: Inconsistencies between adjacent 2D predictions
- **Solution**: Implement 3D consistency constraints and smoothing

### Challenge 4: Computational Efficiency
- **Issue**: Processing large 3D volumes
- **Solution**: Implement batch processing and GPU optimization

## Expected Outcomes
- Accurate kidney and tumor segmentation from CT scans
- Robust performance across different patient demographics
- Fast inference suitable for clinical deployment
- Interpretable results with confidence scores

## Future Enhancements
- Multi-modal integration (CT + MRI)
- Real-time processing capabilities
- Integration with PACS systems
- Clinical validation studies

## Citation
If you use this code, please cite:
```
@article{medsam2023,
  title={MedSAM: A Foundation Model for Medical Image Segmentation},
  author={...},
  journal={...},
  year={2023}
}
```