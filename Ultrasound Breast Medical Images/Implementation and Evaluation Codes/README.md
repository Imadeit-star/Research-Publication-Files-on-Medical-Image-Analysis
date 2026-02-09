
# This repository contains the official research code, datasets and supplementary materials for the study:

## Project Title and Introduction
Project Title: Reliable CNN Evaluation via Variance-Aware Cross-Validation.

# Introduction:
Deep learning models for medical imaging are highly sensitive to data splits and hyperparameter choices.
Traditional single-run or mean-only cross-validation strategies often yield optimistic and unstable performance estimates
This work proposes a Variance-Aware K-Fold Cross-Validation framework
that: - explicitly penalizes fold-to-fold performance variability - promotes stable and generalizable hyperparameter selection - integrates seamlessly with Bayesian Optimization and Tree-structured Parzen Estimator (TPE) methods - is architecture-independent and library-agnostic
The framework is validated on a multi-class breast ultrasound imaging dataset using a modified ResNet-101 backbone.

# Key Contributions:
•	Variance-regularized objective function for hyperparameter optimization
•	Reliability-aware K-Fold evaluation strategy
•	Integration with multiple optimization libraries:
       o	Optuna
       o	Scikit-optimize
       o	GPyOpt
       o	Hyperopt

•	Theoretical generalization error bounds for variance-aware CV
•	Extensive empirical evaluation across multiple K-values



## Key Features:
- **Custom Dataset Handling**: Implementation of a `MedicalImageDataset` class for flexible data loading.

- **Advanced Data Augmentation**: Utilization of both `torchvision.transforms` and `albumentations` for comprehensive image augmentation to enhance model generalization.

- **Transfer Learning**: Employing a pre-trained ResNet101 model on ImageNet for faster convergence and better performance on medical imaging tasks.

- **Fine-tuning**: Selective unfreezing of ResNet101 layers (Layer 3, Layer 4, and the final FC layer) to adapt the model to the specific dataset.

- **Hyperparameter Optimization**: Bayesian optimization via (GPyOpt, Optuna and Scikit) and TPE via (Optuna and Hyperopt) with K-fold cross-validation to find optimal hyperparameters.

- **Dynamic Learning Rate Scheduling**: Support for various learning rate schedulers (StepLR, ReduceLROnPlateau, CosineAnnealingLR, CyclicLR, OneCycleLR) to optimize the training process.

## Installation and Setup
To set up the project environment, ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate # On Windows use `venv\Scripts\activate`

# Install the required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # For CUDA 11.8
Packages and their specific requirements
pip install tensorflow==2.19.0  optuna==4.7.0  matplotlib==3.10.0  scikit-learn==1.6.1  seaborn==0.13.2  numpy==1.26.4   Pillow   torchsummary==1.5.1  albumentations==2.0.8  scikit-optimize==0.10.2  GPyOpt==1.2.6  GPy==1.13.2  hyperopt==0.2.7 ipython==7.34.0
```
**Hardware Requirements**: A GPU (e.g., NVIDIA CUDA enabled) is highly recommended for training due to the computational intensity of the deep learning model.

## Dataset Information
The project utilizes a medical image dataset, structured with subfolders representing different classes  ('Benign', 'Malignant', 'Normal').

Dataset: Breast Ultrasound Images (BUSI)
Total: 777 images

source: Al-Dhabyani et al., Data in Brief (2020)
- **Path**: Specify working directory based on where you place datasets

- **Image Extensions**: `.png`, `.jpg`, `.jpeg`

- **Custom `MedicalImageDataset`**: A custom `MedicalImageDataset` class handles loading images from the specified root directory and mapping class names to numerical labels (`Benign`: 0, `Malignant`: 1, `Normal`: 2).


## Model Architecture
The core of the classification system is a ResNet101 model, pre-trained on ImageNet.

- **Base Model**: `torchvision.models.resnet101(weights='ResNet101_Weights.DEFAULT')`

- **Freezing/Unfreezing**: Most layers are initially frozen (layer1 and layer2). For fine-tuning, `model.layer3`, `model.layer4`, and the final `model.fc` (fully connected) layer are unfrozen.

- **Classifier Modification**: The original `model.fc` layer is replaced with a new `nn.Sequential` block:
    - `nn.Linear(num_ftrs, 3)`: An output layer for 3 classes.
    - `nn.Dropout(dropout_rate)`: A dropout layer for regularization.
    - `nn.Softmax(dim=1)`: Softmax activation for multi-class probability distribution.

- **Loss Function**: `nn.CrossEntropyLoss()` is used, suitable for multi-class classification.

- **Optimizer**: `optim.AdamW` is employed, optimizing only the unfrozen layers with a specified learning rate and weight decay.

## Hyperparameter Optimization

**Optimized Parameters:**
- `learning_rate`: Continuous domain (1e-6, 1e-2)
- `dropout_rate`: Continuous domain (0.05, 0.5)
- `batch_size`: Discrete domain (16, 32, 64)
- `num_epochs`: Discrete domain (10, 101)
- `gamma1`: A continuous parameter (0.001, 10.0) used in the objective function to balance mean accuracy and variance across folds.
- `scheduler_type`: Discrete domain (0, 1, 2, 3, 4) representing different learning rate schedulers.
- **Scheduler-specific parameters**: `factor`, `patience` (for ReduceLROnPlateau), `step_size`, `gamma` (for StepLR), `T_max`, `eta_min` (for CosineAnnealingLR), `base_lr`, `max_lr`, `step_size_up`, `step_size_down`, `mode` (for CyclicLR).

## Training Process
The training process is encapsulated within the `Train_Model` function, which handles both training and validation phases over a specified number of epochs.

- K-Fold Cross-Validation (K = 3–10): Integrated into each method - library pair

- **Phases**: Each epoch consists of a 'train' phase and a 'val' (validation) phase.

- **Model State**: The model is set to `model.train()` for the training phase and `model.eval()` for the validation phase.

- **Optimization**: Gradients are zeroed, outputs are computed, loss is calculated, and backpropagation (`loss.backward()`) followed by `optimizer.step()` is performed only in the training phase.

- **Performance Tracking**: `running_loss` and `running_corrects` are accumulated to calculate `epoch_loss` and `epoch_acc` for both phases.

- **Best Model Saving**: The model weights corresponding to the best validation accuracy are saved.

- **Scheduler Step**: The chosen learning rate scheduler is stepped after each epoch's validation phase, with specific logic depending on the scheduler type (e.g., `ReduceLROnPlateau` steps based on validation accuracy).

# Evaluation Protocol:
 • K-Fold Cross-Validation (K = 3–10)

 •	Metrics reported as Mean ± Standard Deviation:
      o	Accuracy
      o	Precision
      o	Recall
      o	F1-score

•	Per-class and macro-level analysis
•	Comparison against single-run and baselines models



# Rreproducibility:
•	Fixed random seeds for reproducibility

•	Explicit hyperparameter search spaces defined in the code

•	Library-independent optimization pipelines

•	All experiments reported with variance statistics



## How to Run For Reproducibility of Results
To replicate the results or run the project:
1. Clone the repository.

2. Set up the environment as described in the 'Installation and Setup' section.

3. Place your medical image dataset in your path specified. Ensure it's structured with subfolders for each class.

4. Execute the Python script containing the model definition, dataset handling, and training loop. Run the hyperparameter optimization for 50 iterations using K-values from (3-10).

5. Repeat step 4 for each method-library pair eg: Bayesian_GPyOpt_Hyperparameter_Optimization.ipynb etc

6. Evauate best configuration of hyperparameters after the 50 iterations for all K-values using the Evaluation code provided. 
Ensure that you adjust the scheduler parameters within the "Train_Model_Final_KFold" function of the evaluation code accordily for each set of optimized hyperparameters.


## Licensing
This project is licensed under the MIT License - see the `LICENSE` file for details.

## Contact Information
For any questions, suggestions, or collaborations, please contact [Peter Abban; Mehdi Taassori/Obuda University] at [abban@stud.uni-obuda.hu/taassori.mehdi@uni-obuda.hu].
