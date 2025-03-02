# pageflip-cv-detection
# MonReader Page Flip Detection

## Project Overview
MonReader is a mobile document digitization solution that automatically detects when pages are being flipped, allowing for hands-free, high-quality document scanning in bulk. This repository contains an optimized CNN model that can predict if a page is being flipped using a single image frame.

## Problem Statement
For automatic document scanning, we need to detect the precise moment when a user flips a page to capture high-resolution images. The model must accurately distinguish between:
- Image frames showing a page being flipped (positive class)
- Image frames showing static pages (negative class)

## Dataset
- **Total images**: 2,989
- **Training set**: 2,392 images (1,162 flip, 1,230 not flip)
- **Testing set**: 597 images (290 flip, 307 not flip)
- **Data format**: JPG images named as VideoID_FrameNumber.jpg
- **Important note**: The dataset consists of still images (frames) extracted from recordings, not actual video files
- **Frame sequences**: Images are organized by VideoID and frame number to represent sequential moments

## Model Architecture
- **Model type**: Optimized CNN with optional motion feature integration
- **Input size**: 96x96 RGB images
- **Model size**: 1.415 MB (370,177 parameters)
- **Key components**:
  - 3 convolutional blocks with batch normalization
  - Motion detection block
  - Feature fusion for combining visual and motion features
  - Classification head with dropout for regularization

## Preprocessing Pipeline
1. **Basic preprocessing**:
   - Cropping unnecessary background
   - Resizing to target dimensions
   
2. **Full preprocessing** (optional):
   - Cropping unnecessary background
   - Contrast enhancement
   - Sharpening
   - Resizing to target dimensions

## Motion Features
The model can extract motion-related features from sequences of frames:
- Mean motion (average pixel difference between consecutive frames)
- Standard deviation of motion
- Maximum motion value

## Training Approach
- **Optimizer**: Adam with initial learning rate of 0.001
- **Learning rate scheduler**: ReduceLROnPlateau
- **Regularization**: L2 regularization, dropout, and gradient clipping
- **Batch size**: 128
- **Early stopping**: 3 epochs patience
- **Threshold optimization**: 0.15 (determined by maximizing F1 score)

## Results
- **Final metrics** (with threshold=0.15):
  - Accuracy: 96.31%
  - Precision: 96.85%
  - Recall: 95.52%
  - F1 Score: 96.18%
  - Specificity: 97.07%

## Key Findings
1. The optimal classification threshold of 0.15 (rather than the default 0.5) significantly improved performance
2. Motion features calculated from consecutive frames provided valuable information
3. Model size was reduced while maintaining high performance
4. Using regularization techniques improved validation stability
5. The training process completed in approximately 35 minutes

## How to Use the Model
```python
# Load the trained model
model = OptimizedPageFlipNet(include_motion_features=True)
model.load_state_dict(torch.load('optimized_model.pth')['model_state_dict'])
model.eval()

# Preprocess a single image frame
image = preprocess_image(input_image, preprocessing_level='basic')
image_tensor = transform(image).unsqueeze(0)

# Make prediction on a single frame
with torch.no_grad():
    output = model(image_tensor)
    probability = output.item()
    prediction = 'Flip' if probability > 0.15 else 'Not Flip'
```

## Visualizations
The repository includes visualizations of:
- Frame distribution by label
- Preprocessing effects on image frames
- Training metrics
- Threshold optimization
- Confusion matrix
- Example predictions on sample frames

## Requirements
- PyTorch
- NumPy
- OpenCV
- scikit-learn
- PIL
- pandas
- matplotlib
- seaborn

