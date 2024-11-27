# Deep Learning for Computer Vision: CNN, Faster R-CNN, and Vision Transformer (ViT) on MNIST

Explore deep learning techniques for image classification using the MNIST dataset, focusing on CNN, Faster R-CNN, and Vision Transformer (ViT). This lab compares the performance of these models and investigates the benefits of fine-tuning pretrained architectures.

---

## Objective

The primary objective of this lab is to implement and evaluate different deep learning architectures for computer vision tasks, specifically for classifying the MNIST dataset. The key models explored are:

- **Convolutional Neural Networks (CNN)**
- **Faster R-CNN**
- **Vision Transformer (ViT)**

---

## Tasks

### Part 1: CNN Classifier
1. **CNN Model**  
   - Designed a CNN using PyTorch to classify MNIST images.  
   - Included Convolutional, Pooling, and Fully Connected layers.  
   - Defined hyperparameters like kernel size, padding, stride, optimizers, and regularization.

2. **Faster R-CNN Model**  
   - Implemented a Faster R-CNN architecture for MNIST classification.  
   - **Challenge**: Encountered memory usage issues during the Faster R-CNN implementation, which required adjusting batch sizes and model parameters to manage GPU memory effectively.


3. **Comparison**  
   - Compared CNN and Faster R-CNN using metrics such as:  
     - **Accuracy**  
     - **F1 Score**  
     - **Loss**  
     - **Training Time**

4. **Fine-Tuning with Pretrained Models**  
   - Fine-tuned pretrained models (VGG16 and AlexNet) on MNIST.  
   - Analyzed results and compared them with CNN and Faster R-CNN performance.

---


## Tools and Environment

- **PyTorch**: Framework for building and training deep learning models.  
- **Google Colab / Kaggle**: Platforms for running experiments and code.  
- **GitHub**: For version control and project sharing.



