# Experimental Page Flip Detection
This directory contains experimental implementations and extensions of the main page flip detection system.

# GPU-Accelerated Implementation
The notebook Experimental_Page_Flip.ipynb includes a GPU-optimized version of the page flip detection system with the following characteristics:

## GPU-accelerated model (0.10 threshold)

-Training time reduced from 34.45 to 12.15 minutes (nearly 3x faster)
-Accuracy: 94.97%
-F1 Score: 94.62%
-Precision: 98.51%
-Recall: 91.03%



Note: While the final GPU implementation shows a slightly lower recall compared to the optimized CPU model, earlier experiments showed that reducing the threshold generally improved recall across implementations. The final threshold value of 0.10 represents a balance that optimizes overall performance metrics for the GPU implementation.

## Model Threshold Analysis
Through systematic experimentation, I discovered important relationships between implementation details and optimal threshold values:

1. **Original CPU model**: 0.45 threshold (baseline)
2. **Optimized CPU model**: 0.15 threshold (main implementation)
3. **GPU-accelerated model**: 0.10 threshold (experimental version)

This analysis reveals how hardware differences and optimization techniques can fundamentally change model behavior, requiring careful threshold tuning to maintain performance.

## Advanced Visualization Techniques
I implemented GradCAM visualization to interpret model decisions:

- Confirmed the model focuses on relevant features (page edges, book boundaries, hand positions)
- Adapted the visualization approach to work with our architecture for out-of-distribution images
- Generated heatmaps showing high activation areas that correspond to page flip indicators

## Cross-Domain Testing
Conducted experiments with custom cookbook images outside the training distribution:
- Evaluated model generalization capabilities
- Identified limitations for completely different image types
- Gained insights for potential improvements in domain adaptation

## Skills Demonstrated
- GPU optimization for deep learning
- Model interpretability techniques
- Threshold optimization across implementations
- Cross-domain testing and evaluation
- Problem solving for technical implementation challenges
