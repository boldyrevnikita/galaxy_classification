# Galaxy Classification with WaveMix Neural Network Architecture

I implemented a deep learning approach to classify galaxy images from the Galaxy10 DECals dataset. My solution uses the WaveMix architecture with dropout regularization. Here is a breakdown of the methods employed:

## Dataset Exploration and Preparation
I utilized the Hugging Face Datasets library to load the Galaxy10 DECals dataset, which contains 17,736 color galaxy images (256×256 pixels) divided into 10 classes. The dataset was already split into:
• Training set: 15,962 images
• Test set: 1,774 images

I further divided the training set into:
• Training: 13,567 images
• Validation: 2,395 images

The class distribution includes diverse galaxy morphologies from disturbed galaxies to edge-on galaxies with bulge. I created visualization functions to display examples from each class to better understand the dataset.

## Model Architecture: WaveMix
My approach leverages the WaveMix neural network architecture, which is relatively novel for image classification tasks. WaveMix combines wavelet transforms with standard neural network components, making it particularly effective for:
• Capturing multi-scale features in images
• Handling spatial hierarchies common in astronomical data
• Processing high-resolution details while maintaining global context

The WaveMix architecture applies wavelet transformations at different layers, enabling the network to analyze images at multiple frequencies simultaneously, which is advantageous for distinguishing subtle differences between galaxy types.

## Regularization Techniques
I implemented dropout regularization to improve model generalization. This technique:
• Randomly deactivates neurons during training with a specified probability
• Forces the network to learn redundant representations
• Reduces co-adaptation of features
• Helps prevent overfitting, especially important with a relatively small dataset like Galaxy10

I experimented with several different magnitudes of dropout, but perhaps they all were still too aggressive and prevented the model from training well, even though they were aimed at increasing the model's generalization ability.

## Augmentations
To increase the model's generalization ability I included several augmentations:
1. RandomHorizontalFlip: Randomly flips images horizontally with a 50% probability. This is useful for galaxy classification as horizontal orientation generally doesn't change the class of a galaxy.
2. RandomVerticalFlip: Similarly flips images vertically with 50% probability, creating variation in orientation.
3. RandomRotation(30): Rotates images randomly by an angle between -30 and +30 degrees. Since galaxies can appear at any orientation in space, this augmentation helps the model become invariant to rotation.
4. Resize: Scales all images to a consistent square size defined in the CONFIG dictionary, ensuring uniform input dimensions for the neural network.
5. ColorJitter: Randomly alters brightness, contrast, saturation, and hue of images:
• Brightness variation of ±20%
• Contrast variation of ±20%
• Saturation variation of ±20%
• Hue variation of ±10%
This helps the model become robust to different imaging conditions and telescope calibrations.
6. RandomAffine: Applies small geometric transformations including:
• Rotation up to ±10 degrees
• Translation up to 5% in both horizontal and vertical directions
This simulates slight changes in camera perspective or galaxy positioning.
7. ToTensor: Converts PIL images to PyTorch tensors and scales pixel values to1.
8. Normalize: Standardizes pixel values using ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), which helps with model convergence.

## Training Process
The training process includes:
• Batch processing of image data
• Gradient descent optimization (Adam)
• Learning rate scheduling (ReduceLRonPlateau)
• Regular evaluation on validation data

My implementation indicates a methodical approach to hyperparameter tuning, with particular attention to the dropout rate to find the optimal balance between model capacity and generalization.
I also experimented with transformers (SWIN and ViT), but they did not score high
