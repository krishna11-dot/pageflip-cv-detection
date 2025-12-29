# Evaluation Metrics & Results

## Overview

Evaluation is how we measure if our model actually works. This document explains:
1. **What metrics we use** (and why)
2. **How we optimize the decision threshold**
3. **How to interpret results**
4. **Expected performance levels**

---

## Classification Metrics Fundamentals

### The Confusion Matrix

Every prediction falls into one of four categories:

```
                    Predicted
                 Not Flip  |  Flip
              ────────────┼─────────
   Actual     Not Flip│  TN  │  FP  │
              ────────────┼─────────
   Not Flip    Flip   │  FN  │  TP  │
              ────────────┼─────────

TN = True Negative:  Correctly identified not-flip
TP = True Positive:  Correctly identified flip
FP = False Positive: Said flip, but was not-flip (FALSE ALARM)
FN = False Negative: Said not-flip, but was flip (MISSED FLIP)
```

### Real-World Example

```
Ground Truth: [0, 1, 1, 0, 1, 0, 0, 1]  ← Actual labels
Predictions:  [0, 1, 0, 0, 1, 1, 0, 1]  ← Model output

Analysis:
Position 0: Predicted 0, Actual 0 → TN ✓
Position 1: Predicted 1, Actual 1 → TP ✓
Position 2: Predicted 0, Actual 1 → FN ✗ (Missed flip!)
Position 3: Predicted 0, Actual 0 → TN ✓
Position 4: Predicted 1, Actual 1 → TP ✓
Position 5: Predicted 1, Actual 0 → FP ✗ (False alarm!)
Position 6: Predicted 0, Actual 0 → TN ✓
Position 7: Predicted 1, Actual 1 → TP ✓

Confusion Matrix:
           Not Flip | Flip
Not Flip     3      |  1    (FP)
Flip         1      |  3    (TP)
            (FN)
```

---

## Core Metrics

### 1. Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
         = (Correct predictions) / (Total predictions)
```

**Example**:
```
TP=300, TN=800, FP=50, FN=50
Accuracy = (300 + 800) / (300 + 800 + 50 + 50)
         = 1100 / 1200
         = 0.917 (91.7%)
```

**Why It's NOT Enough**:
```
Scenario: Imbalanced dataset
- 900 not-flip frames
- 100 flip frames

Dumb Model: Always predict "not flip"
Accuracy = 900 / 1000 = 90%  ← Looks good!

But: Never detects any flips! ✗
```

**When to Use**: Only when classes are balanced

### 2. Precision

```
Precision = TP / (TP + FP)
          = (Correctly identified flips) / (All predicted flips)
          = "When I say flip, how often am I right?"
```

**Example**:
```
Model says "flip" 100 times
Actually correct 80 times
False alarms 20 times

Precision = 80 / 100 = 0.80 (80%)
```

**Real-World Meaning**:
```
High Precision (0.9):
  • Model is conservative
  • When it says "flip", trust it
  • Use case: Annotating videos (don't want false marks)

Low Precision (0.5):
  • Lots of false alarms
  • Half of "flip" predictions are wrong
  • Use case: Unacceptable for production
```

**Interview Question**: "Why is precision important?"
**Answer**: "Precision measures the cost of false positives. In a page flip detector, high precision means users trust the system's flip markers. Low precision leads to alert fatigue - users stop trusting the system when it cries wolf too often."

### 3. Recall (Sensitivity)

```
Recall = TP / (TP + FN)
       = (Correctly identified flips) / (All actual flips)
       = "Of all flips that happened, how many did I catch?"
```

**Example**:
```
100 actual flips in dataset
Model detects 75 of them
Misses 25 of them

Recall = 75 / 100 = 0.75 (75%)
```

**Real-World Meaning**:
```
High Recall (0.9):
  • Model catches most flips
  • Might have more false alarms
  • Use case: Don't want to miss important page turns

Low Recall (0.5):
  • Misses half of all flips
  • Use case: Unacceptable - defeats purpose of detector
```

**Interview Question**: "Why is recall important?"
**Answer**: "Recall measures the cost of false negatives. For a page flip detector, low recall means missing actual page turns, which defeats the system's purpose. In applications like automatic book scanning or content segmentation, missing flips leads to incomplete results."

### 4. F1 Score (Harmonic Mean)

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = Balanced metric between precision and recall
```

**Why Harmonic Mean, Not Average?**

```
Example: Precision=0.9, Recall=0.5

Arithmetic Mean = (0.9 + 0.5) / 2 = 0.70  ← Misleading!
Harmonic Mean (F1) = 2×(0.9×0.5)/(0.9+0.5) = 0.64  ← More honest

F1 penalizes imbalance more heavily
```

**Intuitive Understanding**:
```
Scenario 1: Balanced
  Precision = 0.85
  Recall = 0.85
  F1 = 2×(0.85×0.85)/(0.85+0.85) = 0.85  ← Same as both!

Scenario 2: Imbalanced
  Precision = 0.95  (High!)
  Recall = 0.50     (Low!)
  F1 = 2×(0.95×0.50)/(0.95+0.50) = 0.66  ← Pulls down!

F1 rewards models that balance both metrics
```

**Interview Question**: "Why F1 as the primary metric?"
**Answer**: "F1 score is ideal because it balances precision and recall. In page flip detection, we need both: high precision ensures user trust (few false alarms) and high recall ensures completeness (catching most flips). F1's harmonic mean penalizes models that sacrifice one metric for the other, encouraging balanced performance."

### 5. Specificity

```
Specificity = TN / (TN + FP)
            = (Correctly identified not-flips) / (All actual not-flips)
            = "Of all non-flip frames, how many did I correctly identify?"
```

**Why It Matters**:
```
High Specificity (0.95):
  • Good at identifying normal frames
  • Few false alarms
  • System won't be noisy

Low Specificity (0.60):
  • Many normal frames labeled as flips
  • System will be very noisy
  • User annoyance
```

### 6. Balanced Accuracy

```
Balanced Accuracy = (Recall + Specificity) / 2
                  = Average of true positive rate and true negative rate
```

**When to Use**: Imbalanced datasets where standard accuracy is misleading

---

## Threshold Optimization

### The Problem with Default Threshold

```python
# Default approach
prediction = 1 if probability > 0.5 else 0
```

**Why 0.5 Might Be Wrong**:

```
Scenario: Imbalanced dataset
- 80% not-flip
- 20% flip

Model learns to be conservative
Typical probabilities:
- Not-flip frames: 0.1 - 0.3
- Flip frames: 0.4 - 0.8

With threshold=0.5:
  Many flip frames (prob=0.4-0.5) classified as not-flip
  Recall suffers!

With threshold=0.3:
  More flips detected
  Better recall!
```

### Threshold Search Algorithm

```python
def find_optimal_threshold(model, val_loader):
    # 1. Collect all predictions
    probabilities = []
    true_labels = []

    for batch in val_loader:
        outputs = model(batch)
        probabilities.extend(outputs)
        true_labels.extend(batch.labels)

    # 2. Test different thresholds
    thresholds = [0.1, 0.15, 0.2, ..., 0.85, 0.9]
    results = []

    for threshold in thresholds:
        predictions = [1 if p > threshold else 0 for p in probabilities]

        precision = calculate_precision(predictions, true_labels)
        recall = calculate_recall(predictions, true_labels)
        f1 = calculate_f1(precision, recall)

        results.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })

    # 3. Select threshold with highest F1
    best = max(results, key=lambda x: x['f1'])
    return best['threshold']
```

### Visualizing Threshold Impact

```
Metrics vs Threshold:
  │
1.0│              Precision ────────
  │                         ╲
  │                          ╲
0.8│                           ╲
  │                            ╲
  │                             ╲
0.6│      ╱─────                 ╲
  │    ╱        F1 ───────────────
0.4│  ╱                            ╲
  │ ╱ Recall                        ╲
0.2│╱                                ─────
  │
  └────────────────────────────────────────>
  0.1   0.3   0.5   0.7   0.9  Threshold

Key Insights:
- Low threshold → High recall, low precision
- High threshold → High precision, low recall
- Optimal threshold → Maximizes F1 (balance)
```

### Trade-offs at Different Thresholds

```
┌─────────────┬───────────┬──────────┬────────┬─────────────┐
│ Threshold   │ Precision │  Recall  │   F1   │  Use Case   │
├─────────────┼───────────┼──────────┼────────┼─────────────┤
│   0.3       │   0.72    │   0.95   │  0.82  │ High recall │
│             │           │          │        │ needed      │
├─────────────┼───────────┼──────────┼────────┼─────────────┤
│   0.42      │   0.85    │   0.87   │  0.86  │ Balanced ✓  │
│             │           │          │        │             │
├─────────────┼───────────┼──────────┼────────┼─────────────┤
│   0.5       │   0.88    │   0.78   │  0.83  │ Default     │
│             │           │          │        │             │
├─────────────┼───────────┼──────────┼────────┼─────────────┤
│   0.7       │   0.95    │   0.65   │  0.77  │ High prec.  │
│             │           │          │        │ needed      │
└─────────────┴───────────┴──────────┴────────┴─────────────┘
```

**Example Decision**:
```
Application: Automatic book scanning
Requirement: Don't miss page turns (high recall priority)

Choice: threshold = 0.3
Result: Catches 95% of flips, accepts some false alarms
Rationale: Better to review a few false positives than miss pages
```

---

## Evaluation Process

### Step 1: Validation Set Evaluation

```python
# During training
for epoch in range(num_epochs):
    train_metrics = train_epoch(...)
    val_metrics = validate_epoch(...)

    print(f"Epoch {epoch}")
    print(f"Train - Loss: {train_metrics['loss']:.4f}, F1: {train_metrics['f1']:.4f}")
    print(f"Val   - Loss: {val_metrics['loss']:.4f}, F1: {val_metrics['f1']:.4f}")
```

**What to Look For**:
```
Good Training:
Epoch 1 - Train Loss: 0.520, Val Loss: 0.548
Epoch 2 - Train Loss: 0.412, Val Loss: 0.435
Epoch 3 - Train Loss: 0.365, Val Loss: 0.380  ← Both decreasing
Epoch 4 - Train Loss: 0.332, Val Loss: 0.358  ← Small gap
Epoch 5 - Train Loss: 0.310, Val Loss: 0.352  ← Converging
✓ Good generalization

Bad Training (Overfitting):
Epoch 1 - Train Loss: 0.520, Val Loss: 0.548
Epoch 2 - Train Loss: 0.412, Val Loss: 0.435
Epoch 3 - Train Loss: 0.320, Val Loss: 0.415  ← Val increasing!
Epoch 4 - Train Loss: 0.250, Val Loss: 0.458  ← Gap widening!
✗ Overfitting detected

Noisy Training (NORMAL):
Epoch 1 - Train F1: 0.52, Val F1: 0.61
Epoch 2 - Train F1: 0.66, Val F1: 0.82  ← Big jump
Epoch 3 - Train F1: 0.74, Val F1: 0.35  ← Massive dip!
Epoch 4 - Train F1: 0.76, Val F1: 0.84  ← Recovered
Epoch 5 - Train F1: 0.80, Val F1: 0.86
✓ Individual dips are NORMAL - focus on overall trend
```

**Key Insight - Training is Noisy**:

Training metrics naturally fluctuate due to:
1. **Stochastic mini-batch sampling**: Different samples each epoch
2. **Dropout randomness**: Different neurons dropped each forward pass
3. **Data augmentation**: Creates different variations
4. **Loss landscape exploration**: Optimizer navigates complex terrain

**What Matters**:
- ✓ Overall trend improving (despite noise)
- ✓ Final convergence to stable plateau
- ✓ Both train and val decreasing together
- ✗ Individual epoch drops (expected noise)

**Real Example - The Epoch 3 Anomaly**:
```
Epoch 3 Validation Metrics:
  Precision: 100% (perfect!)
  Recall:     21% (terrible!)
  F1:         35% (harmonic mean pulls down)

Why?
  Model became extremely conservative temporarily
  Only predicted "flip" when 100% certain
  Missed 79% of actual flips

Is this a problem?
  NO - it recovered by Epoch 4 (F1: 0.84)
  Just noise in the stochastic optimization process

Lesson:
  Don't panic over individual epoch drops
  Look at 3-5 epoch windows for trends
```

**Interview Question**: "How do you handle noisy validation metrics?"

**Answer**: "I focus on trends over 3-5 epoch windows rather than individual epochs. In my training, Epoch 3 had a validation F1 dip to 0.35 due to the model becoming temporarily conservative (100% precision, 21% recall). By Epoch 4 it recovered to 0.84. This taught me that stochastic optimization inherently has noise. I monitor: (1) overall trend direction, (2) both train and val decreasing together, (3) final convergence stability. Individual epoch anomalies are expected and not concerning if the overall trajectory is positive."


### Step 2: Threshold Optimization

```python
optimal_threshold = find_optimal_threshold(model, val_loader)
print(f"Optimal threshold: {optimal_threshold:.2f}")

# Example output
# Threshold 0.42: F1=0.862, Precision=0.851, Recall=0.874
```

### Step 3: Test Set Evaluation

```python
test_metrics = test_with_threshold(
    model,
    test_loader,
    threshold=optimal_threshold
)

print(f"Test Results:")
print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
print(f"  Precision: {test_metrics['precision']:.4f}")
print(f"  Recall:    {test_metrics['recall']:.4f}")
print(f"  F1 Score:  {test_metrics['f1']:.4f}")
```

### Step 4: Confusion Matrix Analysis

```python
plot_confusion_matrix(
    test_metrics['labels'],
    test_metrics['predictions'],
    title="Test Set Confusion Matrix"
)
```

**Example Confusion Matrix**:
```
                 Predicted
              Not Flip | Flip
         ─────────────┼──────────
Actual   Not Flip│ 850  │  45  │ Specificity = 850/895 = 0.95
         ─────────────┼──────────
Not Flip    Flip │  52  │ 365  │ Recall = 365/417 = 0.88
         ─────────────┼──────────

Precision = 365/(365+45) = 0.89
Accuracy = (850+365)/1312 = 0.93
F1 = 2×(0.89×0.88)/(0.89+0.88) = 0.88
```

**Interpreting the Matrix**:
- **TN=850**: Correctly identified 850 non-flip frames
- **FP=45**: Falsely flagged 45 non-flip frames as flips (5% false alarm rate)
- **FN=52**: Missed 52 actual flips (12% miss rate)
- **TP=365**: Correctly detected 365 flips

**Interview Question**: "What does the confusion matrix tell you?"
**Answer**: "The confusion matrix provides granular insight into model behavior. In this case:
- High TN (850) shows good specificity - we correctly identify most normal frames
- Low FP (45) confirms high precision - few false alarms
- Moderate FN (52) shows room for improvement in recall
- High TP (365) confirms the model catches most flips

The matrix reveals that when the model fails, it's more likely to miss a flip (FN=52) than falsely flag one (FP=45), suggesting we could lower the threshold slightly to catch more flips."

---

## Expected Results

### Target Metrics

```
┌─────────────────┬─────────┬──────────────────────────────┐
│     Metric      │ Target  │         Interpretation       │
├─────────────────┼─────────┼──────────────────────────────┤
│  F1 Score       │ > 0.85  │ Good balance, production-ready│
│  Accuracy       │ > 0.90  │ Overall correctness          │
│  Precision      │ > 0.80  │ User trust (few false alarms)│
│  Recall         │ > 0.80  │ Completeness (catch most)    │
│  Specificity    │ > 0.90  │ Low false alarm rate         │
└─────────────────┴─────────┴──────────────────────────────┘
```

### Performance by Data Characteristic

```
┌──────────────────────┬─────────────────────────────────┐
│   Data Split         │     Expected F1 Score           │
├──────────────────────┼─────────────────────────────────┤
│  Training Set        │  0.90 - 0.95  (Should be high)  │
│  Validation Set      │  0.85 - 0.90  (Slightly lower)  │
│  Test Set            │  0.82 - 0.88  (Final measure)   │
└──────────────────────┴─────────────────────────────────┘

┌──────────────────────┬─────────────────────────────────┐
│   Frame Position     │     Expected Performance        │
├──────────────────────┼─────────────────────────────────┤
│  Beginning (0-9)     │  Lower (fewer flips in data)    │
│  Middle (10-19)      │  Higher (most action here)      │
│  End (20+)           │  Lower (fewer samples)          │
└──────────────────────┴─────────────────────────────────┘
```

---

## Visualization & Analysis

### Training History Plot

```python
plot_training_history(history, "Page Flip Model")
```

**What to Look For**:

1. **Loss Curves**:
```
   Loss
    │
0.6 │╲╲
    │  ╲╲           Train ──
0.4 │    ╲╲╲        Val ----
    │      ╲╲╲╲____
0.2 │          ────────
    │
    └──────────────────────> Epoch
    1    3    5    7    9

✓ Both decreasing
✓ Gap stays small
✓ Both stabilize
```

2. **Metric Curves (with Real Training Noise)**:
```
   F1
    │
0.9 │            ╱────────  ← Stable plateau
    │          ╱
0.7 │     ╱──╱  ↓ Epoch 3 dip (normal noise!)
    │   ╱
0.5 │ ╱
    │╱
    └──────────────────────> Epoch
    1    3    5    7    9

✓ Overall trend improving (despite Epoch 3 dip)
✓ Reaches stable plateau
✓ Individual fluctuations are NORMAL
```

**Real Training Curves - What We Actually Saw**:

```
Epoch-by-Epoch Metrics:

Epoch  | Train Loss | Val Loss | Train F1 | Val F1  | Notes
───────┼────────────┼──────────┼──────────┼─────────┼──────────────────
1      | 0.79       | 0.67     | 0.52     | 0.61    | Initial
2      | 0.52       | 0.46     | 0.66     | 0.82    | Big improvement
3      | 0.42       | 0.68     | 0.74     | 0.35    | ⚠ Val dip (noise!)
4      | 0.37       | 0.41     | 0.76     | 0.84    | Recovered
5      | 0.33       | 0.36     | 0.80     | 0.86    | Steady improvement
6      | 0.31       | 0.29     | 0.82     | 0.87    |
7      | 0.29       | 0.25     | 0.84     | 0.88    |
8      | 0.28       | 0.22     | 0.85     | 0.89    |
9      | 0.27       | 0.18     | 0.87     | 0.89    |
10     | 0.26       | 0.15     | 0.89     | 0.90    | Final

Key Observations:
✓ Overall trend: Both losses decrease, F1 increases
✓ Epoch 3 anomaly: Temporary, recovered immediately
✓ Val sometimes better than train: Normal (dropout, augmentation)
✓ Final metrics: Train=0.89, Val=0.90 (excellent balance)
```

**Interview Insight**: "My training visualizations show the typical noise in stochastic optimization. The Epoch 3 validation F1 dip to 0.35 might look alarming in isolation, but examining the full trajectory shows steady improvement with temporary fluctuations. This reinforces focusing on trends rather than individual data points. See [Visualization Analysis](09_visualization_analysis_and_interview_questions.md) for detailed breakdown."

### Random Sample Testing

```python
test_on_random_images(
    model,
    test_df,
    transform,
    num_samples=5,
    threshold=optimal_threshold
)
```

**Expected Output**:
```
Sample 1:
  Original → Preprocessed → Prediction
  [Image]     [Image]       Not Flip: 0.12 ✓
                            Threshold: 0.42

Sample 2:
  Original → Preprocessed → Prediction
  [Image]     [Image]       Flip: 0.89 ✓
                            Threshold: 0.42

...
```

**What to Check**:
- ✓ Confident correct predictions (prob far from threshold)
- ⚠ Close calls (prob near threshold) - acceptable in small numbers
- ✗ Confident wrong predictions (prob far from threshold, but wrong) - investigate!

---

## Error Analysis

### When Does the Model Fail?

#### Common Failure Modes

1. **Motion Blur Extremes**
```
Very Fast Flip:
  Entire image is blur → Hard to distinguish from camera shake
  Solution: More training data with fast flips

Very Slow Flip:
  Minimal motion → Looks like static frame
  Solution: Lower threshold or better motion features
```

2. **Occlusions**
```
Hand Covering Page:
  Page edge not visible → Model confused
  Solution: Train with more varied hand positions
```

3. **Lighting Changes**
```
Flash During Flip:
  Sudden brightness change → False positive
  Solution: Better brightness augmentation
```

4. **Similar Motions**
```
Turning Book (Not Flipping Page):
  Similar motion pattern → False positive
  Solution: More diverse negative samples
```

### Analyzing Misclassifications

```python
# Find worst mistakes
test_results = test_with_threshold(model, test_loader, threshold)

# High confidence errors
wrong_predictions = test_results['predictions'] != test_results['labels']
high_confidence = abs(test_results['outputs'] - 0.5) > 0.4
worst_errors = wrong_predictions & high_confidence

print(f"High-confidence errors: {worst_errors.sum()}")
# Inspect these carefully!
```

---

## Success Criteria Summary

### Model is "Good" if:

1. ✓ **F1 Score > 0.85**: Balanced precision and recall
2. ✓ **Val/Train gap < 10%**: Good generalization
3. ✓ **Confusion matrix balanced**: No extreme bias
4. ✓ **Stable training**: Smooth convergence, no oscillation
5. ✓ **Reasonable confidence**: Not all predictions near threshold

### Model is "Excellent" if:

1. ⭐ **F1 Score > 0.90**: Very strong performance
2. ⭐ **Works on varied data**: Different videos, lighting, speeds
3. ⭐ **Fast inference**: < 50ms per frame
4. ⭐ **Interpretable**: Can explain why it predicts flip/not-flip

---

## Interview Preparation

### Key Points to Remember

1. **Metric Selection**:
   - "I use F1 as primary metric because it balances precision and recall, both critical for this task"
   - "Accuracy alone would be misleading due to class imbalance"

2. **Threshold Optimization**:
   - "Default 0.5 threshold is arbitrary - I optimize based on validation F1"
   - "This allows adapting to class imbalance and application requirements"

3. **Trade-offs**:
   - "Higher threshold → fewer false positives, more missed flips"
   - "Lower threshold → catch more flips, more false alarms"
   - "I chose the threshold that maximizes F1, balancing both"

4. **Limitations**:
   - "Model struggles with extreme motion blur and occlusions"
   - "Performance degrades on videos very different from training data"
   - "Future work: Add LSTM for temporal modeling, use ensemble methods"

---

## Next Steps

- Read [Project Overview](01_project_overview.md) for context
- Read [Architecture Documentation](02_architecture.md) for model design
- Read [Training Strategy](04_training_strategy.md) for optimization details
- See main [README](../README.md) for quick start
