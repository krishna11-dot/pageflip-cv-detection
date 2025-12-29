# Data Pipeline Documentation

## Overview

The data pipeline transforms raw video frames into model-ready inputs with both spatial and temporal features.

```
Raw Images → Motion Features → Preprocessing → Augmentation → Model Input
   (JPG)         (Temporal)      (Spatial)      (Training)     (Tensors)
```

---

## Phase 1: Dataset Creation

### Directory Structure

```
images/
├── training/
│   ├── flip/              ← Frames containing page flips
│   │   ├── video1_05.jpg
│   │   ├── video1_15.jpg
│   │   └── ...
│   └── notflip/           ← Frames without flips
│       ├── video1_01.jpg
│       ├── video1_02.jpg
│       └── ...
└── testing/
    ├── flip/
    └── notflip/
```

### Filename Convention

```
video1_05.jpg
  │    │   │
  │    │   └─ Extension (.jpg)
  │    └───── Frame number (05)
  └────────── Video ID (video1)
```

**Why This Matters**:
- Video ID groups frames from same sequence
- Frame number enables temporal ordering
- Essential for calculating motion between consecutive frames

### Dataset DataFrame Structure

```python
create_dataset_df(base_path) → DataFrame
```

**Output**:
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `image_path` | str | Full path to image | `/path/video1_05.jpg` |
| `label` | int | 0=notflip, 1=flip | 1 |
| `video_id` | str | Video identifier | `video1` |
| `frame_number` | int | Frame position | 5 |
| `split` | str | training/testing | `training` |
| `sequence_position` | str | beginning/middle/end | `beginning` |

**Sequence Position Logic**:
```
Frame 0-9:   "beginning"  ← Video start, less likely to have flips
Frame 10-19: "middle"     ← Most action happens here
Frame 20+:   "end"        ← Video conclusion
```

### Dataset Statistics (analyze_dataset)

```python
analyze_dataset(df)
```

**What It Checks**:

1. **Class Balance**
   ```
   Total images: 5,240
   Training: 3,928 (75%)
   Testing:  1,312 (25%)

   Class distribution:
   NotFlip: 3,500 (67%)
   Flip:    1,740 (33%)
   ```

   **Why This Matters**:
   - Imbalanced classes can bias the model
   - If 90% are "notflip", model might just predict "notflip" always
   - We need to monitor this and potentially use class weights

2. **Video-Level Distribution**
   ```
   Unique videos in training: 45
   Unique videos in testing:  15
   ```

   **Why This Matters**:
   - Train and test videos MUST be different (no leakage!)
   - If same video appears in train and test → inflated accuracy

3. **Per-Video Class Balance**
   ```
   video1: NotFlip=25, Flip=5 (5:1 ratio)
   video2: NotFlip=28, Flip=2 (14:1 ratio) ← Highly imbalanced!
   ```

   **Why This Matters**:
   - Some videos may have very few flip frames
   - Model might struggle to learn from imbalanced videos

### Success Criteria for Dataset
- ✓ No missing files or corrupted images
- ✓ Reasonable class balance (ideally 30-70% flip frames)
- ✓ Train/test video separation (no overlap)
- ✓ Sufficient samples per class (>500 each)

---

## Phase 2: Motion Feature Extraction

### The Core Problem

**Question**: How do we capture the MOTION of a page flip?

**Answer**: Compare consecutive frames to detect changes.

### Motion Extraction Algorithm

```python
extract_optimized_motion_features(current_frame, previous_frame) → [3 features]
```

#### Step-by-Step Process

```
Input: Two consecutive frames
┌─────────────┐      ┌─────────────┐
│ Frame t-1   │      │ Frame t     │
│             │      │             │
│    📄       │      │   📄 →      │  (Page moving)
│             │      │             │
└─────────────┘      └─────────────┘

Step 1: Convert to Grayscale
┌─────────────┐      ┌─────────────┐
│ 480×640×3   │  →   │ 480×640×1   │
│ (RGB)       │      │ (Grayscale) │
└─────────────┘      └─────────────┘

Why grayscale?
• Motion patterns visible without color
• 3× faster processing (1 channel vs 3)
• Reduces memory usage

Step 2: Resize to 64×64
┌─────────────┐      ┌─────────┐
│ 480×640     │  →   │ 64×64   │
│             │      │         │
└─────────────┘      └─────────┘

Why resize?
• Motion patterns visible at lower resolution
• ~56× fewer pixels (409,600 → 4,096)
• MUCH faster computation
• Trade-off: Lose fine details, but motion is still clear

Step 3: Calculate Frame Difference
difference = abs(current - previous)

Example:
Current:    Previous:   Difference:
┌─────┐     ┌─────┐     ┌─────┐
│ 200 │  -  │ 180 │  =  │ 20  │
│ 150 │     │ 150 │     │  0  │
│ 100 │     │ 120 │     │ 20  │
└─────┘     └─────┘     └─────┘

Step 4: Extract Statistics

difference = [
  [20, 0, 15, 5, ...],   ← Row 1
  [0, 0, 25, 30, ...],   ← Row 2
  [10, 5, 0, 0, ...],    ← Row 3
  ...
]

Feature 1: mean_motion = mean(difference)
  = average pixel change across entire frame

  High mean → Lots of movement (flip likely)
  Low mean  → Little movement (no flip)

  Example: mean_motion = 12.5

Feature 2: std_motion = std(difference)
  = variability of motion across frame

  High std → Non-uniform motion (page edge moving)
  Low std  → Uniform motion (camera shake)

  Example: std_motion = 24.3

Feature 3: max_motion = max(difference)
  = maximum pixel change anywhere

  High max → Sharp, localized motion (flip edge)
  Low max  → Gentle, distributed motion

  Example: max_motion = 87.0
```

### Why These 3 Features?

#### 1. Mean Motion (Overall Activity)
```
No Flip:                Page Flip:
┌───────────┐          ┌───────────┐
│           │          │█▓▒░       │
│           │          │█▓▒░       │  ← Significant
│           │          │█▓▒░       │     change
│           │          │           │
└───────────┘          └───────────┘
mean ≈ 2.5             mean ≈ 25.3
```

**Interview Question**: "Why use mean?"
**Answer**: "Mean motion quantifies overall activity. Page flips involve significant pixel changes, resulting in higher mean values compared to static frames."

#### 2. Standard Deviation (Motion Uniformity)
```
Camera Shake:           Page Flip:
┌───────────┐          ┌───────────┐
│▒▒▒▒▒▒▒▒▒▒▒│          │███        │  ← Non-uniform
│▒▒▒▒▒▒▒▒▒▒▒│          │██▒        │     (edge moves,
│▒▒▒▒▒▒▒▒▒▒▒│          │█░░        │      center static)
│▒▒▒▒▒▒▒▒▒▒▒│          │░          │
└───────────┘          └───────────┘
std ≈ 3.2              std ≈ 28.7
(uniform)              (non-uniform)
```

**Interview Question**: "Why standard deviation?"
**Answer**: "Std captures motion distribution. Page flips have non-uniform motion (edges move more than center), while camera shake or global motion is more uniform. High std indicates localized motion characteristic of flips."

#### 3. Maximum Motion (Peak Intensity)
```
Slow Movement:          Page Flip:
┌───────────┐          ┌───────────┐
│░░░░       │          │███        │  ← Sharp edge
│░░░░       │          │███        │     = high max
│           │          │           │
│           │          │           │
└───────────┘          └───────────┘
max ≈ 15               max ≈ 187
```

**Interview Question**: "Why max motion?"
**Answer**: "Max captures peak intensity. Page flips create sharp edges between the flipping page and background, resulting in high local contrast changes. This spike in maximum difference is a strong flip indicator."

### Parallel Processing with Caching

```python
calculate_optimized_motion_features(df, use_cache=True)
```

**Optimization Strategy**:

```
Without Optimization:           With Optimization:
────────────────────           ────────────────────
Process videos sequentially    Process videos in parallel
Frame 1 → Frame 2 → Frame 3   Frame 1 ┐
  ↓         ↓         ↓                ├→ Parallel
Video 1   Video 2   Video 3   Frame 2 ┤
                                       │
Time: ~30 minutes                Frame 3┘

                               Time: ~5 minutes

                               Cache to disk:
                               motion_features_cache.npz

                               Next run: Load from cache
                               Time: ~10 seconds ✓
```

**Why Caching Matters**:
- Motion calculation is expensive (read images, compute diffs)
- Features don't change between runs
- Cache enables fast experimentation with model architectures

---

## Phase 3: Image Preprocessing

### Three Preprocessing Levels

```python
preprocess_image(image, preprocessing_level='basic')
```

#### Level 1: None (Baseline)
```
┌─────────────────┐
│  Original Image │
│      480×640    │
└─────────────────┘
        ↓
    Resize only
        ↓
┌─────────────────┐
│     96×96       │
└─────────────────┘
```
**Use Case**: Baseline comparison, fastest processing

#### Level 2: Basic (Default)
```
┌─────────────────────┐
│  Original Image     │
│  ┌──────────────┐   │
│  │              │   │  ← Extra background
│  │   Content    │   │
│  │              │   │
│  └──────────────┘   │
└─────────────────────┘
        ↓
    Crop background
        ↓
┌──────────────┐
│   Content    │  ← Focused on relevant area
└──────────────┘
        ↓
    Resize to 96×96
        ↓
┌──────────────┐
│    96×96     │
└──────────────┘
```
**Use Case**: Default for training, good balance

#### Level 3: Full (Maximum Enhancement)
```
Original → Crop → Enhance → Sharpen → Resize
                    ↓         ↓
                 Contrast   Edges
                   ×1.2     ×1.1
```
**Use Case**: When image quality is poor, experimental

### Normalization (Critical Step!)

```python
# Apply to ALL images
transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],  # RGB channels
    std=[0.229, 0.224, 0.225]
)
```

**What This Does**:
```
Before Normalization:
pixel_value = 150 (range: 0-255)

After Normalization:
pixel_value = (150/255 - 0.485) / 0.229
            = (0.588 - 0.485) / 0.229
            = 0.450

Result: Values roughly in range [-2, +2]
```

**Why These Specific Values?**
- ImageNet dataset statistics (standard in computer vision)
- Neural networks train better with normalized inputs
- Prevents certain channels from dominating

**Interview Question**: "Why normalize?"
**Answer**:
1. **Gradient stability**: Large pixel values (0-255) → large gradients → unstable training
2. **Zero-centered**: Helps with weight initialization and convergence
3. **Standard practice**: Using ImageNet stats enables transfer learning later

---

## Phase 4: Data Augmentation (Training Only)

### Why Augment?

**Problem**: Limited training data → Model memorizes training set
**Solution**: Create variations to improve generalization

```python
class PageFlipDataset(Dataset):
    def __init__(self, ..., augment=True):
        self.augment = augment
```

### Augmentation Techniques

#### 1. Random Rotation (±5 degrees)
```
Original:              Augmented:
┌──────────┐          ┌──────────┐
│   📄     │    →     │  📄      │  (Rotated 3°)
│          │          │          │
└──────────┘          └──────────┘

Why small angles?
• Page flip videos naturally have slight camera angles
• Too much rotation (>10°) would be unrealistic
• Helps model generalize to different camera positions
```

#### 2. Random Brightness (0.95× to 1.05×)
```
Original:              Augmented:
┌──────────┐          ┌──────────┐
│  ■■■■    │    →     │  ▓▓▓▓    │  (Slightly darker)
│  ■■■■    │          │  ▓▓▓▓    │
└──────────┘          └──────────┘

Why subtle changes?
• Different lighting conditions in videos
• Flash, shadows, ambient light variations
• Too much change would distort features
```

#### 3. Color Jitter
```python
transforms.ColorJitter(brightness=0.05, contrast=0.05)
```

**Why ONLY During Training?**
```
Training Set:                 Validation/Test Set:
Apply augmentation            NO augmentation
↓                            ↓
Model sees variations        Evaluate on clean data
↓                            ↓
Learns robust features       Measure true performance
```

**Interview Question**: "Why not augment validation/test data?"
**Answer**: "Augmentation is for improving generalization during training. For evaluation, we need consistent, unmodified data to measure true model performance. Augmenting test data would give artificially inflated metrics."

---

## Phase 5: DataLoader Configuration

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=128,        # Large batch for stability
    shuffle=True,          # Randomize order
    num_workers=4,         # Parallel data loading
    pin_memory=True,       # Faster GPU transfer
    persistent_workers=True # Keep workers alive
)
```

### Batch Size: Why 128?

```
Small Batch (32):           Large Batch (128):
────────────────           ─────────────────
Noisy gradients            Smoother gradients
Longer training            Faster training
Less memory               More memory
More updates per epoch     Fewer updates per epoch

Trade-off:
32  → Better generalization, slower
128 → Faster training, stable gradients ✓
256 → Too large for our dataset, might underfit
```

**Our Choice**: 128 is a sweet spot
- Fast training (fewer iterations)
- Stable gradient estimates
- Fits in memory (96×96 images are small)

### Shuffle: Why True for Training?

```
Without Shuffle:             With Shuffle:
────────────────            ───────────────
Batch 1: video1_01-08      Batch 1: video3_15, video1_02, ...
Batch 2: video1_09-16      Batch 2: video2_04, video5_20, ...
Batch 3: video1_17-24      Batch 3: video1_08, video4_11, ...

Problem: Sequential frames   Solution: Random mix
are very similar            ↓
↓                          Model sees diverse examples
Model might overfit to      each batch
specific video sequences    ↓
                           Better generalization ✓
```

### Num Workers: Why 4?

```
num_workers=0:               num_workers=4:
──────────────              ─────────────
Main Process                Main Process + 4 Workers
   ↓                           ↓
Load Batch 1               Worker 1: Load Batch 1
   ↓                       Worker 2: Load Batch 2
Train on Batch 1           Worker 3: Load Batch 3
   ↓                       Worker 4: Load Batch 4
Load Batch 2                   ↓
   ↓                       Main: Train on Batch 1
Train on Batch 2                ↓
   ↓                       Main: Train on Batch 2 (already loaded!)
...
                           Result: GPU never waits for data
Time: ~20 min              Time: ~8 min ✓
```

### Pin Memory: Why True?

```
Without pin_memory:          With pin_memory:
───────────────────         ────────────────
CPU Memory (Pageable)       CPU Memory (Pinned)
      ↓                            ↓
Copy to Staging Area        Direct Transfer
      ↓                            ↓
Transfer to GPU             Transfer to GPU

Time: ~15ms per batch       Time: ~5ms per batch ✓
```

**Trade-off**: Uses slightly more RAM, but much faster GPU transfer

---

## Data Pipeline Success Criteria

### 1. Data Quality Checks
```python
# Check for issues
assert len(df) > 1000, "Dataset too small"
assert df['label'].value_counts()[0] / len(df) < 0.9, "Too imbalanced"
assert df.isna().sum().sum() == 0, "Missing values detected"
```

### 2. Motion Feature Validation
```python
# Visualize motion features by class
plt.hist(df[df['label']==0]['mean_motion'], alpha=0.5, label='Not Flip')
plt.hist(df[df['label']==1]['mean_motion'], alpha=0.5, label='Flip')
```

**Expected Result**: Flip frames should have higher motion values
```
     Not Flip ░░░░░░
     Flip     ████████░░░░

     0     10    20    30    40
          mean_motion →
```

### 3. Preprocessing Validation
```python
visualize_preprocessing(df, num_samples=3)
```

**Check**:
- ✓ Images are properly cropped (no excessive background)
- ✓ Normalized values in reasonable range
- ✓ No artifacts or distortions

### 4. DataLoader Performance
```python
# Measure loading speed
start = time.time()
for batch in train_loader:
    pass  # Just loading
print(f"Time: {time.time() - start:.2f}s")
```

**Target**: < 5 seconds to iterate through entire dataset

---

## Common Issues and Solutions

### Issue 1: Motion features all zeros
```python
# Symptom
df['mean_motion'].describe()
# mean: 0.0, std: 0.0

# Cause: Missing previous frames
# Solution: Check video_id grouping
```

### Issue 2: Out of memory during training
```python
# Symptom
RuntimeError: CUDA out of memory

# Solutions:
1. Reduce batch_size: 128 → 64
2. Reduce image_size: 96 → 64
3. Reduce num_workers: 4 → 2
```

### Issue 3: Slow data loading
```python
# Symptom
GPU utilization < 50%

# Causes & Solutions:
1. num_workers=0 → Increase to 4
2. No pin_memory → Enable pin_memory=True
3. No persistent_workers → Enable for faster restarts
```

---

## Next Steps

- Read [Training Strategy Documentation](04_training_strategy.md) for optimization techniques
- Read [Evaluation Metrics Documentation](05_evaluation_metrics.md) for performance analysis
- See [Architecture Documentation](02_architecture.md) for how data flows through the model
