# Project Overview: Page Flip Detection System

## Purpose: Why This Project?

This project implements a **real-time page flip detection system** for video frames. The system can automatically identify when a page is being turned in a video sequence.

### Real-World Applications

1. **Digital Libraries & E-Readers**: Automatically detect page turns in book scanning videos
2. **Video Annotation**: Mark key moments in educational or documentary content
3. **Accessibility Tools**: Help visually impaired users navigate scanned documents
4. **Content Indexing**: Automatically segment video content by page boundaries

## The Core Challenge

Detecting a page flip is not trivial because:

- **Motion blur**: Pages move quickly, creating blur that obscures details
- **Varied lighting**: Shadows and lighting changes during the flip
- **Partial visibility**: Pages are only partially visible during the flip action
- **Background noise**: Camera shake, hand movements, other objects moving
- **Speed variation**: People flip pages at different speeds

## Why This Approach?

### Decision 1: Why Use Both Images AND Motion Features?

**The Problem**: Using images alone misses critical temporal information.

**The Solution**: Combine spatial features (from images) with temporal features (from motion)

```
Image Features (CNN)          Motion Features (Frame Diff)
        │                              │
        │                              │
        └──────────┬───────────────────┘
                   │
           Feature Fusion Layer
                   │
            Classification
```

**Why This Works**:
- Images capture WHAT is in the frame (book, hand, page shape)
- Motion captures HOW things are changing between frames
- A page flip has both distinctive appearance AND distinctive motion patterns
- Combining both provides more robust detection

### Decision 2: Why Deep Learning Instead of Traditional CV?

**Alternatives Considered**:
- ❌ **Edge detection + heuristics**: Too brittle, fails with lighting changes
- ❌ **Optical flow + thresholds**: Sensitive to parameters, doesn't generalize
- ✅ **CNN + Motion Features**: Learns patterns automatically, robust to variations

**Why CNN**:
- Automatically learns hierarchical features (edges → shapes → patterns)
- Robust to lighting, angle, and appearance variations
- Can handle complex, non-linear relationships in data
- Generalizes well to unseen videos

### Decision 3: Why This Specific Architecture?

**Multi-Scale Feature Extraction**:
- Small kernels (3×3): Detect fine edges and details
- Medium kernels (5×5): Capture broader motion patterns
- Different scales → More comprehensive understanding

**Regularization Strategy**:
- BatchNorm: Stabilizes training, reduces internal covariate shift
- Dropout: Prevents overfitting by randomly dropping connections
- L2 Regularization: Keeps weights small, improves generalization
- Early Stopping: Stops before overfitting begins

**Why This Matters**: We want the model to work on NEW videos, not just memorize the training data.

## Success Criteria

### What Does "Success" Mean for This Project?

#### Primary Metric: **F1 Score** (Balance between Precision and Recall)

**Why F1 and not just Accuracy?**

Imagine a dataset with 90% "not flip" and 10% "flip" frames:
- A model that always predicts "not flip" would get 90% accuracy!
- But it would be completely useless (never detects actual flips)

**F1 Score** forces the model to:
- Have high **Precision**: When it says "flip", it's usually correct
- Have high **Recall**: It catches most actual flips, not just a few

#### Secondary Metrics:

1. **Precision**: "Of all frames we marked as flip, how many were actually flips?"
   - Important for avoiding false alarms
   - High precision → User trust

2. **Recall**: "Of all actual flip frames, how many did we detect?"
   - Important for not missing flips
   - High recall → Complete coverage

3. **Specificity**: "How good are we at identifying non-flip frames?"
   - Prevents false positives from overwhelming the system

#### Model Performance Goals:

| Metric | Target | Why This Target? |
|--------|--------|------------------|
| **F1 Score** | > 0.85 | Good balance for production use |
| **Precision** | > 0.80 | Avoid annoying false positives |
| **Recall** | > 0.80 | Don't miss important flips |
| **Training Time** | < 30 min | Practical for experimentation |
| **Inference Speed** | < 50ms/frame | Real-time capability |

## Key Design Decisions Summary

| Decision | Why? | What Success Looks Like |
|----------|------|-------------------------|
| **Use Motion Features** | Page flips have distinctive temporal patterns | Motion features correlate with flip events |
| **Image Size: 96×96** | Balance between detail and speed | Fast training without sacrificing accuracy |
| **Varied Kernel Sizes** | Capture features at multiple scales | Better feature extraction than uniform kernels |
| **Optimal Threshold** | Handle class imbalance | Higher F1 than default 0.5 threshold |
| **Data Caching** | Motion calculation is expensive | 10-20× faster on subsequent runs |
| **Regularization** | Prevent overfitting | Val loss close to train loss |

## What Makes This Project "Good"?

### Technical Excellence:

1. **Clear Reasoning**: Every choice (motion features, kernel sizes, threshold) has a documented reason
2. **Performance Optimization**: Caching, multiprocessing, batch size tuning
3. **Proper Validation**: Train/val/test split, early stopping, optimal threshold search
4. **Comprehensive Metrics**: Not just accuracy - full confusion matrix analysis

### Demonstrable Understanding:

1. **Know the "Why"**: Can explain why motion features help, why we need regularization
2. **Know the Trade-offs**: Smaller images → faster but less detail, higher dropout → less overfitting but might underfit
3. **Know the Limitations**: Works best on clear page flip videos, might struggle with very fast/slow flips
4. **Know the Alternatives**: Considered and rejected simpler approaches with valid reasons

## Interview-Ready Talking Points

### "Why did you use two models?"
**WRONG**: "I wanted to try something different"
**RIGHT**: "I'm not using two separate models. I'm using one CNN model that processes two input streams: spatial features from images and temporal features from frame differences. This multi-modal approach captures both WHAT is in the frame and HOW it's changing, which is essential for detecting the dynamic action of a page flip."

### "Why these specific motion features?"
**WRONG**: "They seemed to work"
**RIGHT**: "I chose three motion statistics from frame differencing:
- **mean_motion**: Captures overall activity level - flips have higher average motion
- **std_motion**: Captures motion variability - flips have non-uniform motion (edges move more than center)
- **max_motion**: Captures peak intensity - flips have sharp, localized motion at edges

These three metrics efficiently summarize the temporal signature of a flip without requiring expensive optical flow computation."

### "What would you improve if you had more time?"
**HONEST ANSWER**:
"Three things:
1. **Temporal modeling**: Add LSTM to model sequences, not just single frames
2. **Class weighting**: If flip frames are rare, weight them more in the loss function
3. **Transfer learning**: Start with pretrained ResNet instead of training from scratch - would likely improve accuracy with less data"

## Next Steps

After understanding this overview:
1. Read [Architecture Documentation](02_architecture.md) for system design details
2. Read [Data Pipeline](03_data_pipeline.md) for data processing workflow
3. Read [Training Strategy](04_training_strategy.md) for optimization techniques
4. Read [Results](06_results.md) for performance analysis
