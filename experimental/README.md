# Experimental Page Flip Detection

This directory contains experimental implementations and extensions of the main page flip detection system.

## GPU-Accelerated Implementation

The notebook `Experimental_Page_Flip.ipynb` includes a GPU-optimized version of the page flip detection system with the following characteristics:

- **GPU-accelerated model (0.10 threshold)**
  - Training time reduced from 34.45 to 12.15 minutes (nearly 3x faster)
  - Accuracy: 94.97%
  - F1 Score: 94.62%
  - Precision: 98.51%
  - Recall: 91.03%

Note: While the final GPU implementation shows a slightly lower recall compared to the optimized CPU model, earlier experiments showed that reducing the threshold generally improved recall across implementations. The final threshold value of 0.10 represents a balance that optimizes overall performance metrics for the GPU implementation.

## Threshold Evolution Analysis

A key finding across implementations has been the significant shift in optimal threshold values:

1. **Original CPU model**: 0.45 threshold
2. **Optimized CPU model**: 0.15 threshold (main implementation)
3. **GPU-accelerated model**: 0.10 threshold

This threshold shift demonstrates how implementation details and hardware differences can significantly affect model behavior.

## Additional Experiments

### Custom Image Testing
Tests on out-of-distribution cookbook images to evaluate the model's cross-domain generalization capabilities.

### GradCAM Visualization
Implementation of a simplified GradCAM to visualize feature activations, showing that the model focuses on:
- Page edges
- Book boundaries
- Hand positions

The visualizations confirmed these elements are key for detecting page flips.

## Technical Challenges

When implementing GradCAM, dimension mismatches were encountered between the feature extraction and classifier parts of the model when processing custom images. A simplified approach was implemented that visualizes feature activations from earlier layers.
