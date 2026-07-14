# Cat vs Dog Classification

## Project Overview

This repository contains a complete TensorFlow-based deep learning pipeline for classifying cat and dog images using a convolutional neural network (CNN). The project is designed to prepare the dataset, clean corrupted images, perform data augmentation, train a robust CNN model, and generate evaluation metrics and visualizations for model performance.

## Key Features

- Data cleaning for corrupted or empty image files
- Train/validation split with `image_dataset_from_directory`
- Data augmentation using horizontal flip, rotation, zoom, and contrast adjustments
- Image normalization and performance optimization with `tf.data`
- CNN architecture with convolutional, batch normalization, pooling, and dropout layers
- Early stopping to avoid overfitting
- Evaluation with classification report, confusion matrix, ROC curve, and AUC score
- Visualization outputs for accuracy, loss, confusion matrix, ROC curve, and sample images
- Model export to `model/cats_vs_dogs_cnn.keras`

## Dataset

This project uses the Microsoft Cats vs Dogs dataset from Kaggle:

https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset

The dataset is intentionally excluded from this repository. Download it manually and organize it into the following directory structure:

```
dataset/
  Cat/
    cat.0.jpg
    cat.1.jpg
    ...
  Dog/
    dog.0.jpg
    dog.1.jpg
    ...
```

## Repository Structure

- `main.py` - The main training and evaluation script
- `README.md` - Project documentation
- `LICENSE.md` - License and usage terms
- `Visualization/` - Example saved figures produced by the training script
- `model/` - Saved model output directory created at runtime
- `.gitignore` - Git ignore rules for virtual environments and local dataset files

## Requirements

The project uses the following Python packages:

- Python 3.8+ (recommended)
- TensorFlow
- NumPy
- Pillow
- matplotlib
- seaborn
- scikit-learn

## Setup Instructions

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd "Cat vs Dog Classification"
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install tensorflow numpy pillow matplotlib seaborn scikit-learn
   ```

4. Download the Kaggle Cats vs Dogs dataset and place the extracted `Cat` and `Dog` folders under `dataset/`.

## Running the Project

Run the main training script:

```bash
python main.py
```

The script performs the following steps:

1. Removes corrupted or empty files from the dataset
2. Loads training and validation data from `dataset/`
3. Applies data augmentation and normalization
4. Builds and trains a CNN model
5. Evaluates the model on validation data
6. Saves performance plots and the trained model

## Output Files

The script saves outputs to the following locations:

- `visualization/Sample_Images.png`
- `visualization/Accuracy.png`
- `visualization/Loss.png`
- `visualization/Confusion_Matrix.png`
- `visualization/ROC_Curve.png`
- `model/cats_vs_dogs_cnn.keras`

## Notes

- The dataset folder is excluded from version control to protect dataset privacy and comply with usage restrictions.
- If you run this project headlessly on a server, you may need to configure Matplotlib to use a non-interactive backend before execution.
- The model uses a binary classification output layer with sigmoid activation and binary crossentropy loss.

## Contact

Moiz Baloch

- Email: khanmoaiz682@gmail.com
- Phone: +92 306 7892235

## License and Attribution

This project is proprietary and may not be used without explicit consent from the repository owner. Users granted permission to use this project must cite the original owner and this repository in any derivative work or publication.
