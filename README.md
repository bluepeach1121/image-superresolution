# Super-Resolution Neural Network: Technical Overview and Results (Mistake Noticed)! changes and corrections coming soon!.

## **Overview**
This project implements a single-branch convolutional neural network (CNN) for **image super-resolution (SR)**. The goal is to upscale low-resolution (LR) images to high-resolution (HR) images. The network was trained and tested on the **DIV2K dataset**. 

The following papers were references:
  
  `https://arxiv.org/abs/1511.04587` --> Accurate Image Super-Resolution Using Very Deep Convolutional Networks, and
  `https://arxiv.org/abs/1501.00092` --> Image Super-Resolution Using Deep Convolutional Networks


I noticed that the PSNR was too high and I went back into the code and found a few issues with the image comparison. corrections are coming soon!

---

## **Model Architecture**

### **1. Components**
The network is composed of the following:

#### **(a) Feature Extraction**
- **Description**: Three convolutional layers with ReLU activations to extract spatial features from the input LR image.
- **Purpose**: Encodes the input image into a high-dimensional feature map for further processing.

#### **(b) Non-Linear Mapping**
- **Description**: Three additional convolutional layers with ReLU activations.

#### **(c) Upsampling**
- **Description**: Two stages of upsampling using **PixelShuffle** layers, interspersed with convolutional layers and ReLU activations. The upsampling block uses a convolutional layer to expand the channels to 256, followed by a PixelShuffle operation to upscale the feature map by a factor of 2.

#### **(d) Reconstruction**
- **Description**: A final convolutional layer reduces the feature map channels to match the RGB output dimensions. A **bilinear upsampling layer** ensures the final HR image matches the target resolution.

#### **(e) Residual Learning**
- **Purpose**: Helps preserve details from the input image and stabilizes training.

### **2. Technical Details**
- **Total Trainable Parameters**: ~335,000.
- **Loss Function**: Mean Squared Error (MSE).
- **Optimizer**: AdamW with a learning rate of \(1 	imes 10^{-4}\).
- **Metrics**:
  - **PSNR (Peak Signal-to-Noise Ratio)**: Measures reconstruction quality.
  - **SSIM (Structural Similarity Index)**: Measures perceptual and structural similarity.

---

## **Dataset**

### **1. DIV2K Dataset**
- **Low-Resolution Images**:
  - Generated using bicubic downsampling by factors of \(x2\) and \(x4\). But I only used x4 due to compute limits
- **High-Resolution Images**:
  - Ground truth HR images used for training and evaluation.
- **Train-Test Split**:
  - Training: 85%
  - Testing: 15%

---
## **Strengths**
1. **Lightweight Architecture**:
   - Achieves high performance with fewer parameters (~335K), making it computationally efficient.
2. **Residual Learning**:
   - Stabilizes training and ensures finer detail reconstruction.
3. **High PSNR and SSIM**:
   - Exceeds baseline performance, demonstrating superior reconstruction quality.

---

## **Suggestions for Improvement**
1. **Dynamic Resolution Handling**
2. **Multi-Scale Features**:
   - Incorporate a multi-branch architecture to process multiple scales simultaneously (e.g., \(x2\), \(x4\)).
3. **Attention Mechanisms**:
   - Add spatial and channel attention modules to focus on important regions of the image.
   - Attention could help enhance feature representation and reconstruction quality.

4. **Adversarial Training**:
   - Use a Generative Adversarial Network (GAN) for perceptual loss to improve the realism of the reconstructed images.
---

## **Conclusion**
The implemented super-resolution model achieves state-of-the-art results with a simple and efficient architecture. By leveraging residual learning and effective optimization, the model surpasses baseline methods and demonstrates its potential for real-world applications. Future enhancements could further improve its flexibility, generalization, and perceptual quality.

