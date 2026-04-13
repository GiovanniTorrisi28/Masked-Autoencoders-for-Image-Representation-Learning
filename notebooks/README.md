# Jupyter Notebooks

Use this folder only for initial data exploration, quick visualizations, and proof of concepts. 
Actual training code must reside in the Python files inside `src/`.

---

## Notebooks for ImageNet100 & MAE Project

### 1. `01_eda_imagenet100.ipynb`
**Purpose**: Exploratory Data Analysis
- Load and explore train/val splits
- Visualize sample images from random classes
- Calculate class and image statistics
- Check image size distribution and aspect ratios
- Generate reference plots for dataset understanding

**Output**: Saved plots in `figures/01_*`

### 2. `02_patch_visualization.ipynb`
**Purpose**: Understanding MAE Preprocessing
- Demonstrate patch extraction (patchification)
- Visualize how images are divided into 16×16 patches
- Show random masking pattern (75% default for MAE)
- Illustrate masked vs visible patches
- Explain the self-supervised learning objective

**Key Concepts**:
- Image size: 224×224 → 14×14 patches = 196 total patches
- Mask ratio: 75% masked, 25% visible
- Encoder processes visible patches only
- Decoder reconstructs masked patches

**Output**: Saved plots in `figures/02_*`

### 3. `03_dataset_validation.ipynb`
**Purpose**: Data Quality & Integrity Checks
- Validate train/val split consistency
- Detect and report corrupt or missing images
- Check class balance across splits
- Verify image size ranges
- Generate validation report JSON

**Output**: 
- Plots in `figures/03_*`
- Report: `data/processed/imagenet100_validation_report.json`

---

## Workflow

1. **First Run**: Execute `01_eda_imagenet100.ipynb` after downloading dataset
   - Understand data overview
   - Ensure dataset structure is correct

2. **Before Training**: Execute `02_patch_visualization.ipynb`
   - Understand patch-based preprocessing
   - Verify MAE concepts

3. **Validation**: Execute `03_dataset_validation.ipynb`
   - Check dataset health
   - Ensure no corrupt images
   - Confirm train/val consistency

---

## Notes

- These notebooks assume dataset is in `data/raw/imagenet100/`
- Train and val subdirectories with class folders expected
- Labels.json (optional) maps class IDs to descriptions
- All plots saved to `figures/` for inclusion in final report
