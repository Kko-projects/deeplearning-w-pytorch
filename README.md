# Deep Learning with PyTorch 🧠🔥

This project is a curated collection of PyTorch-based deep learning experiments and implementations.  
It covers fundamental operations as well as model training pipelines on datasets like MNIST and CIFAR-10.  
Models include fully-connected networks (MLP), convolutional networks (VGG, ResNet18), and efficient architectures like MobileNetV2.


### 🧠 Core Concepts
- 01 ~ 04

### 🔢 Classification Models
- 05 ~ 08
- 09: Transfer learning with pretrained models (VGG16)

### 🖼️ Object Detection Experiments
This section contains object detection experiments using pre-trained models such as:

- 10: NVIDIA SSD (Single Shot Multibox Detector)
- 🔜 YOLOv5 (Coming soon)
- 🔜 Faster R-CNN (Coming soon)


## 📘 Notebooks

| No. | Notebook | Description |
|-----|----------|-------------|
| 01  | `01_tensor_basics.ipynb` | Tensor creation and basic operations |
| 02  | `02_tensor_gpu_autograd.ipynb` | Using GPU and automatic differentiation |
| 03  | `03_nn_module_and_forward.ipynb` | Custom model definition using nn.Module |
| 04  | `04_loss_and_optimizer.ipynb` | Loss functions and optimizers |
| 05  | `05_binary_classification_circles.ipynb` | Nonlinear classification with synthetic data |
| 06  | `06_mnist_training_and_confusion.ipynb` | MNIST training and confusion matrix evaluation |
| 07  | `07_mnist_batchnorm_models.ipynb` | BatchNorm applied to fully connected and conv models |
| 08  | `08a_cifar10_MLP.ipynb`         | CIFAR-10 classification using MLP model              |
| 08  | `08b_cifar10_VGG.ipynb`         | CIFAR-10 classification using VGG model              |
| 09  | `08c_cifar10_ResNet18.ipynb`     | CIFAR-10 classification using ResNet18 with custom implementation |
| 10  | `08d_cifar10_MobileNet.ipynb`    | CIFAR-10 classification using MobileNetV2 (custom built) |
| 11  | `09_Transfer_Learning.ipynb`     | Flower classification using transfer learning with VGG16 |
| 12  | `10a_SSD_object_detection.ipynb` | SSD object detection using NVIDIA TorchHub and COCO images |



## 🔧 Tech Stack

- Python (PyTorch, torchvision, torch.nn, torch.optim)
- Jupyter / Google Colab
- NumPy, Pandas, Matplotlib, Seaborn
- scikit-learn (for evaluation reports & confusion matrix)


## 🗂 Datasets

This project uses:
- **MNIST**: 28x28 grayscale digits for basic classification tasks.
- **CIFAR-10**: 32x32 RGB images from 10 categories (e.g., airplane, cat, truck).
- **Flowers Dataset (from Udacity)**: ~4,000 color images in 5 classes: `daisy`, `dandelion`, `roses`, `sunflowers`, and `tulips`.

All datasets are automatically downloaded via `torchvision.datasets`.
