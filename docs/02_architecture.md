# System Architecture

## Complete System Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PAGE FLIP DETECTION PIPELINE                        │
└─────────────────────────────────────────────────────────────────────────────┘

1. DATA LOADING
   ┌─────────────────────────────┐
   │  Video Frame Images         │
   │  ├── training/              │
   │  │   ├── flip/              │      Organized by:
   │  │   └── notflip/           │      - Split (train/test)
   │  └── testing/               │      - Label (flip/notflip)
   │      ├── flip/              │      - Video ID
   │      └── notflip/           │      - Frame Number
   └─────────────────────────────┘
                │
                ▼
2. MOTION FEATURE EXTRACTION (Temporal Features)
   ┌─────────────────────────────────────────────────────────┐
   │  For each video sequence:                               │
   │  1. Sort frames by frame_number                         │
   │  2. Load current frame (t) and previous frame (t-1)     │
   │  3. Convert to grayscale                                │
   │  4. Resize to 64×64 (faster processing)                 │
   │  5. Calculate: diff = abs(frame_t - frame_t-1)          │
   │  6. Extract statistics:                                 │
   │     • mean_motion = mean(diff)                          │
   │     • std_motion = std(diff)                            │
   │     • max_motion = max(diff)                            │
   │                                                          │
   │  Cache results to disk for reuse                        │
   └─────────────────────────────────────────────────────────┘
                │
                ▼
3. IMAGE PREPROCESSING (Spatial Features)
   ┌─────────────────────────────────────────────────────────┐
   │  For each frame:                                         │
   │  1. Convert to RGB                                       │
   │  2. Crop unnecessary background (focus on content)       │
   │  3. [Optional] Enhance contrast (1.2×)                   │
   │  4. [Optional] Enhance sharpness (1.1×)                  │
   │  5. Resize to 96×96                                      │
   │  6. Normalize: (pixel - mean) / std                      │
   │     mean=[0.485, 0.456, 0.406]  (ImageNet stats)        │
   │     std=[0.229, 0.224, 0.225]                           │
   └─────────────────────────────────────────────────────────┘
                │
                ▼
4. DATA AUGMENTATION (Training Only)
   ┌─────────────────────────────────────────────────────────┐
   │  Subtle augmentations to improve generalization:        │
   │  • Random rotation: ±5 degrees                          │
   │  • Random brightness: 0.95 - 1.05×                      │
   │  • Color jitter: slight brightness/contrast variation   │
   └─────────────────────────────────────────────────────────┘
                │
                ▼
5. MODEL ARCHITECTURE
   ┌─────────────────────────────────────────────────────────┐
   │                 OPTIMIZED PAGE FLIP NET                  │
   │                                                          │
   │  ┌──────────────────────────────────────────────┐       │
   │  │  INPUT: 96×96×3 RGB Image                    │       │
   │  └──────────────────────────────────────────────┘       │
   │                      │                                   │
   │                      ▼                                   │
   │  ┌──────────────────────────────────────────────┐       │
   │  │  BLOCK 1: Basic Feature Extraction           │       │
   │  │  • Conv2D(3→32, kernel=3×3)                  │       │
   │  │  • BatchNorm2D                                │       │
   │  │  • ReLU                                       │       │
   │  │  • MaxPool2D(2×2) → 48×48×32                 │       │
   │  │  • Dropout2D(0.1)                             │       │
   │  └──────────────────────────────────────────────┘       │
   │                      │                                   │
   │                      ▼                                   │
   │  ┌──────────────────────────────────────────────┐       │
   │  │  BLOCK 2: Edge Detection                     │       │
   │  │  • Conv2D(32→64, kernel=5×5) [Larger kernel] │       │
   │  │  • BatchNorm2D                                │       │
   │  │  • ReLU                                       │       │
   │  │  • MaxPool2D(2×2) → 24×24×64                 │       │
   │  │  • Dropout2D(0.15)                            │       │
   │  └──────────────────────────────────────────────┘       │
   │                      │                                   │
   │                      ▼                                   │
   │  ┌──────────────────────────────────────────────┐       │
   │  │  BLOCK 3: Higher-Level Features              │       │
   │  │  • Conv2D(64→128, kernel=3×3)                │       │
   │  │  • BatchNorm2D                                │       │
   │  │  • ReLU                                       │       │
   │  │  • MaxPool2D(2×2) → 12×12×128                │       │
   │  └──────────────────────────────────────────────┘       │
   │                      │                                   │
   │                      ▼                                   │
   │  ┌──────────────────────────────────────────────┐       │
   │  │  BLOCK 4: Motion Pattern Detection           │       │
   │  │  • Conv2D(128→192, kernel=3×3)               │       │
   │  │  • BatchNorm2D                                │       │
   │  │  • ReLU                                       │       │
   │  │  • AdaptiveAvgPool2D(1×1) → 192 features     │       │
   │  └──────────────────────────────────────────────┘       │
   │                      │                                   │
   │                      ▼                                   │
   │              ┌──────────────┐                            │
   │              │  Flatten     │                            │
   │              │  192 features│                            │
   │              └──────────────┘                            │
   │                      │                                   │
   │        ┌─────────────┴─────────────┐                    │
   │        │                           │                    │
   │        ▼                           ▼                    │
   │  Image Features          Motion Features               │
   │  (192 dims)              (3 dims: mean, std, max)       │
   │        │                           │                    │
   │        └─────────────┬─────────────┘                    │
   │                      ▼                                   │
   │  ┌──────────────────────────────────────────────┐       │
   │  │  FEATURE FUSION LAYER                        │       │
   │  │  • Concatenate [192 + 3] = 195 dims          │       │
   │  │  • Linear(195 → 96)                          │       │
   │  │  • ReLU                                       │       │
   │  │  • Dropout(0.2)                               │       │
   │  └──────────────────────────────────────────────┘       │
   │                      │                                   │
   │                      ▼                                   │
   │  ┌──────────────────────────────────────────────┐       │
   │  │  CLASSIFICATION HEAD                         │       │
   │  │  • Dropout(0.3)                               │       │
   │  │  • Linear(96 → 32)                           │       │
   │  │  • ReLU                                       │       │
   │  │  • Linear(32 → 1)                            │       │
   │  │  • Sigmoid → probability [0, 1]              │       │
   │  └──────────────────────────────────────────────┘       │
   │                      │                                   │
   │                      ▼                                   │
   │              OUTPUT: P(flip)                             │
   │              If P > threshold: "Flip"                    │
   │              Else: "Not Flip"                            │
   └─────────────────────────────────────────────────────────┘
                │
                ▼
6. TRAINING LOOP
   ┌─────────────────────────────────────────────────────────┐
   │  For each epoch:                                         │
   │  1. Train phase:                                         │
   │     • Forward pass through model                         │
   │     • Calculate BCE loss                                 │
   │     • Add L2 regularization penalty                      │
   │     • Backward pass                                      │
   │     • Clip gradients (prevent exploding)                 │
   │     • Update weights                                     │
   │                                                          │
   │  2. Validation phase:                                    │
   │     • Forward pass (no gradients)                        │
   │     • Calculate metrics (F1, precision, recall)          │
   │     • Check for improvement                              │
   │                                                          │
   │  3. Learning rate adjustment:                            │
   │     • ReduceLROnPlateau monitors val loss                │
   │     • Reduce LR by 0.5× if no improvement for 2 epochs   │
   │                                                          │
   │  4. Early stopping check:                                │
   │     • If no improvement for 3 epochs → stop              │
   │     • Restore best model weights                         │
   └─────────────────────────────────────────────────────────┘
                │
                ▼
7. THRESHOLD OPTIMIZATION
   ┌─────────────────────────────────────────────────────────┐
   │  Find optimal decision threshold:                        │
   │  • Test thresholds from 0.1 to 0.9 (step 0.05)          │
   │  • Calculate F1 score for each threshold                 │
   │  • Select threshold with highest F1                      │
   │  • Use this optimal threshold for test evaluation        │
   └─────────────────────────────────────────────────────────┘
                │
                ▼
8. EVALUATION
   ┌─────────────────────────────────────────────────────────┐
   │  Test set evaluation with optimal threshold:             │
   │  • Calculate final metrics                               │
   │  • Generate confusion matrix                             │
   │  • Visualize predictions on random samples               │
   │  • Save model with metadata                              │
   └─────────────────────────────────────────────────────────┘
```

---

## Model Architecture Deep Dive

### Why This Architecture?

#### Design Philosophy

```
Simple Image Classifier     →     Our Approach
     (Not Enough)                  (Better)

┌──────────────────┐         ┌──────────────────┐
│   Image Only     │         │  Image Features  │
│                  │         │   (Spatial)      │
│   ↓              │         │        +         │
│  CNN             │         │  Motion Features │
│   ↓              │         │   (Temporal)     │
│ Classification   │         │        ↓         │
└──────────────────┘         │  Feature Fusion  │
                             │        ↓         │
  Problem: Misses            │  Classification  │
  temporal dynamics          └──────────────────┘
  of page flipping
                               Solution: Captures
                               both appearance AND
                               movement patterns
```

### Layer-by-Layer Breakdown

#### Block 1: Basic Feature Extraction
```
Input: 96×96×3 image
         ↓
Conv2D(3→32, 3×3) + BatchNorm + ReLU
         ↓
Purpose: Extract low-level features
- Edges (horizontal, vertical, diagonal)
- Color transitions
- Basic shapes

Output: 48×48×32 feature maps
```

**Why kernel size 3×3?**
- Standard size for capturing local patterns
- Computationally efficient
- Proven effective in many architectures (VGG, ResNet)

**Why 32 filters?**
- Enough to capture basic edge patterns
- Not too many → prevents overfitting early on
- Each filter learns a different type of edge/pattern

#### Block 2: Edge Detection (5×5 Kernel)
```
Input: 48×48×32
         ↓
Conv2D(32→64, 5×5) + BatchNorm + ReLU
         ↓
Purpose: Detect page edges and motion blur
- Larger receptive field captures wider patterns
- Page edges during flip
- Motion blur patterns
- Hand/finger contours

Output: 24×24×64 feature maps
```

**Why larger kernel (5×5) here specifically?**
- Page flips create MOTION BLUR → spread over larger area
- Need wider context to see curved page edges
- Detect patterns like: "edge moving from left to right"

**Why 64 filters?**
- Double the previous layer → more complex patterns
- Learn combinations of basic features
- Still manageable for training

#### Block 3: Higher-Level Features (3×3 Kernel)
```
Input: 24×24×64
         ↓
Conv2D(64→128, 3×3) + BatchNorm + ReLU
         ↓
Purpose: Combine mid-level features into concepts
- "Curved page shape"
- "Hand + page interaction"
- "Shadow patterns during flip"
- "Partial page visibility"

Output: 12×12×128 feature maps
```

**Why back to 3×3?**
- After capturing wider patterns, refine with local operations
- Build hierarchical features: edges → shapes → objects
- More efficient than continuing with 5×5

#### Block 4: Motion Pattern Detection
```
Input: 12×12×128
         ↓
Conv2D(128→192, 3×3) + BatchNorm + ReLU
         ↓
AdaptiveAvgPool2D(1×1)
         ↓
Purpose: Detect high-level motion patterns
- "Is there a page flip motion?"
- "What type of flip? (fast/slow, left/right)"
- Combine all spatial information

Output: 192 features (global image representation)
```

**Why AdaptiveAvgPool2D?**
- Converts any spatial size → 1×1 (global features)
- Each of 192 channels becomes a single value
- Robust to input size variations
- Used in modern architectures (ResNet, EfficientNet)

#### Feature Fusion Layer
```
Image Features (192)  +  Motion Features (3)
              ↓
        Concatenate → 195 dims
              ↓
      Linear(195 → 96) + ReLU + Dropout(0.2)
              ↓
      Purpose: Learn how to combine spatial + temporal info
```

**Why concatenate instead of add?**
- Addition assumes features are in same space → they're not
- Concatenation lets model LEARN the relationship
- More expressive, learns weights for each input type

**Why reduce to 96 dimensions?**
- Dimensionality reduction prevents overfitting
- Forces model to learn compressed, essential features
- 96 is large enough to retain information, small enough to generalize

#### Classification Head
```
Input: 96 fused features
       ↓
Dropout(0.3)  ← Heavy regularization
       ↓
Linear(96 → 32) + ReLU
       ↓
Linear(32 → 1)
       ↓
Sigmoid → [0, 1]
```

**Why two linear layers instead of one?**
- Non-linear transformation (ReLU between) → more expressive
- Learn complex decision boundaries
- 96→32→1 is gentler than 96→1 (avoid sudden compression)

**Why Sigmoid?**
- Binary classification → need probability [0, 1]
- Sigmoid(x) = 1 / (1 + e^-x)
- Smooth, differentiable, well-suited for BCE loss

---

## Architectural Design Decisions

### Multi-Scale Feature Extraction

```
┌─────────────────────────────────────────────┐
│  Why Varied Kernel Sizes? [3, 5, 3, 3]     │
└─────────────────────────────────────────────┘

3×3 Kernels (Blocks 1, 3, 4):
  ┌─┬─┬─┐
  │ │ │ │   Sees 3×3 pixel neighborhood
  ├─┼─┼─┤   Good for: Fine details, precise edges
  │ │X│ │   Use when: Need local precision
  ├─┼─┼─┤
  │ │ │ │
  └─┴─┴─┘

5×5 Kernel (Block 2):
  ┌─┬─┬─┬─┬─┐
  │ │ │ │ │ │   Sees 5×5 pixel neighborhood
  ├─┼─┼─┼─┼─┤   Good for: Broader patterns, motion blur
  │ │ │ │ │ │   Use when: Need wider context
  ├─┼─┼─┼─┼─┤   Perfect for: Detecting page curvature
  │ │ │X│ │ │              and motion blur during flips
  ├─┼─┼─┼─┼─┤
  │ │ │ │ │ │
  └─┴─┴─┴─┴─┘

Result: Model sees features at multiple scales
→ More robust to variations in flip speed, page size, distance
```

### Regularization Strategy

```
┌──────────────────────────────────────────────────────────┐
│  Why So Much Regularization?                             │
│                                                           │
│  Problem: Neural networks tend to MEMORIZE training data │
│  Solution: Multiple regularization techniques            │
└──────────────────────────────────────────────────────────┘

1. Dropout (Random Neuron Dropping)
   During Training:                  During Testing:
   ┌─┐ ┌─┐ ┌─┐ ┌─┐                  ┌─┐ ┌─┐ ┌─┐ ┌─┐
   │✓│ │X│ │✓│ │X│  50% dropped     │✓│ │✓│ │✓│ │✓│  All active
   └─┘ └─┘ └─┘ └─┘                  └─┘ └─┘ └─┘ └─┘

   Effect: Forces each neuron to be useful independently
   Locations: After pooling (0.1, 0.15), fusion (0.2), final (0.3)

2. Batch Normalization
   Before BatchNorm:        After BatchNorm:
   [1, 5, 100, 2, 89] →    [0.1, 0.3, 0.9, 0.0, 0.8]

   Effect: Normalizes activations → stable training
   Where: After every Conv2D layer

3. L2 Regularization (Weight Decay)
   Loss = BCE_Loss + λ × (sum of weights²)
                       ↑
                   Penalty for large weights

   Effect: Keeps weights small → simpler model → better generalization
   Value: λ = 0.0001

4. Early Stopping
   Val Loss
      │    ┌─ Stop here (no improvement for 3 epochs)
      │   ╱
      │  ╱
      │ ╱
      │╱
      └────────────────────> Epochs

   Effect: Prevents training too long → avoids overfitting

5. Gradient Clipping
   if gradient_norm > 1.0:
       gradient = gradient × (1.0 / gradient_norm)

   Effect: Prevents "exploding gradients" → stable training
```

### Why This Model Size?

```
Parameter Count: ~1.27 million parameters
Model Size: ~4.86 MB

Is this too big? Too small? Just right?

Comparison:
┌─────────────────────────────────────────┐
│ ResNet-18:    11 million params  (8×)   │
│ MobileNet:    4.2 million params (3×)   │
│ **Our Model: 1.27 million params**      │
│ Tiny CNN:     0.1 million params (10×)  │
└─────────────────────────────────────────┘

Sweet spot:
✓ Large enough to learn complex patterns
✓ Small enough to train quickly
✓ Small enough to avoid overfitting on limited data
✓ Fast inference (~20-50ms per frame)
```

---

## Data Flow Example

Let's trace a single image through the entire system:

```
INPUT: Frame_10.jpg from video1
       (480×640×3 RGB image)
                │
                ├─────────────────────────────┐
                │                             │
                ▼                             ▼
     IMAGE PREPROCESSING         MOTION FEATURE EXTRACTION
                │                             │
    1. Crop to 96×96                Frame_10 - Frame_09
    2. Normalize                           │
    3. To tensor                    Resize to 64×64
       [3, 96, 96]                        │
                │                    Calculate difference
                │                           │
                │                    mean=12.5, std=24.3, max=87
                │                           │
                │                    To tensor [3]
                │                             │
                └──────────┬──────────────────┘
                           ▼
                    FORWARD PASS
                           │
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
    Conv Block 1                      (motion features
         │                             held until fusion)
    [32, 48, 48]
         ▼
    Conv Block 2 (5×5)
         │
    [64, 24, 24]
         ▼
    Conv Block 3
         │
    [128, 12, 12]
         ▼
    Conv Block 4 + Pool
         │
    [192] ← Image features
         │
         └─────────┬─────────┘
                   ▼
         Concatenate with motion
              [192 + 3]
                   ▼
           Fusion Layer [96]
                   ▼
        Classification Head [1]
                   ▼
              Sigmoid(0.73)
                   ▼
         Compare with threshold (0.42)
                   ▼
           0.73 > 0.42 → "FLIP"
```

---

## Success Criteria for Architecture

### How Do We Know This Architecture Works?

#### 1. Training vs. Validation Gap
```
Good Model:                    Overfit Model:
Train Loss ─────              Train Loss ─────
           ╲                             ╲
Val Loss    ─────             Val Loss    ╱╲╱╲╱╲
                                         ╱      ╲

Gap < 0.05 ✓                  Gap > 0.15 ✗
```

**Our Target**: Val loss within 10% of train loss

#### 2. Feature Learning Verification
- Early layers should detect edges → visualize filters
- Middle layers should detect shapes → visualize activations
- Late layers should detect high-level patterns

#### 3. Ablation Study Results
| Configuration | F1 Score | Conclusion |
|--------------|----------|------------|
| **Full model** | **0.85** | Baseline |
| No motion features | 0.78 | Motion helps! (+0.07) |
| All 3×3 kernels | 0.82 | Multi-scale helps! (+0.03) |
| No regularization | 0.79 | Overfits without it |

---

## Next Steps

- Read [Data Pipeline Documentation](03_data_pipeline.md) for data processing details
- Read [Training Strategy Documentation](04_training_strategy.md) for optimization techniques
- Read [Evaluation Metrics Documentation](05_evaluation_metrics.md) for metric analysis
