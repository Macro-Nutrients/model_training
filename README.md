# MacroNutrient - Food Classification Model

## Project Overview
This repository contains a deep learning model for food image classification as part of the MacroNutrient Capstone project. The model can classify food images into 5 categories:
- ayam_goreng (fried chicken)
- burger
- donat (donut)
- kentang_goreng (french fries)
- mie_goreng (fried noodles)

## Repository Structure
```
├── inference.ipynb          # Notebook for model inference
├── train.ipynb              # Notebook for model training
├── requirements.txt         # Required dependencies
├── laporan.md               # Detailed project documentation in Indonesian
├── img_for_inference/       # Sample images for inference
└── kaggle/working/
    ├── models/              # Saved model files
    │   ├── saved_model.keras
    │   ├── saved_model/     # TensorFlow SavedModel format
    │   └── tflite/          # TensorFlow Lite model
    │       ├── label.txt
    │       └── model_tf.tflite
    ├── splitted_dataset/    # Training, validation and test datasets
    ├── confusionmatrix_plt.png  # Evaluation visualization
    └── model_evaluation_plt.png # Model performance visualization
```

## Model Architecture
The model uses transfer learning with MobileNetV2 as the base model and includes:
- MobileNetV2 pretrained on ImageNet (frozen layers)
- Custom classification head with convolutional layers, dropout, and dense layers
- Output layer with 5 classes

## Dataset
The dataset consists of food images collected from multiple Kaggle sources:
- [Portuguese Meals Dataset](https://www.kaggle.com/datasets/catarinaantelo/portuguese-meals)
- [Food Classification Dataset](https://www.kaggle.com/datasets/rizkyyk/dataset-food-classification)
- [Fast Food Classification Dataset](https://www.kaggle.com/datasets/utkarshsaxenadn/fast-food-classification-dataset)

Subsequently, the researcher performed sorting and cropping on each image using an image processing application to ensure that only high-quality images were used for training. The finalized dataset was then uploaded to Google Drive and can be accessed via the following link:
- [Final Dataset](https://drive.google.com/file/d/1WSKVHCrDwPnqFau175P5ehI-h4Glog1o/view?usp=sharing)


The data was split as follows:
- Training: 80% (398 images)
- Validation: 10% (49 images)
- Testing: 10% (53 images)

## Training Process
The training process included:
- Image preprocessing with augmentation (rotation, horizontal flip)
- Normalization (rescaling pixel values to [0,1])
- Transfer learning using MobileNetV2
- Model evaluation using accuracy metrics and confusion matrices

## Usage

### Requirements
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Training
To train the model, run:
```bash
jupyter notebook train.ipynb
```

### Inference
To use the model for inference:
```bash
jupyter notebook inference.ipynb
```

## Model Performance
The model achieved high accuracy in classifying the five food categories. Detailed evaluation metrics and visualizations are available in the training notebook and the report document.

## Deployment
The model has been converted to TensorFlow Lite format for mobile deployment. The TFLite model and corresponding label file are available in the tflite directory.

## Project Documentation
For a comprehensive explanation of the project (in Indonesian), please see the detailed report.