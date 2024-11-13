# Lab 1: Deep Learning with PyTorch - Classification and Regression

This repository contains two primary projects focused on using the PyTorch library to perform classification and regression tasks. The main goal of this lab is to establish deep learning proficiency by developing Deep Neural Network (DNN) architectures for these tasks.

## Objective

The primary objective is to gain hands-on experience in creating and fine-tuning DNN architectures using PyTorch, applying these models to two datasets to handle regression and classification problems.

## Datasets

1. **NYSE Stock Dataset**: Available on Kaggle: [New York Stock Exchange Data](https://www.kaggle.com/datasets/dgawlik/nyse). This dataset contains historical stock prices and fundamental indicators of companies on the NYSE, which we use for regression tasks.
   
2. **Predictive Maintenance Dataset**: Available on Kaggle: [Machine Predictive Maintenance Classification](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification). This dataset includes sensor data for predictive maintenance and is used for multi-class classification.

## Project Workflow

### Part 1: Regression on NYSE Dataset

1. **Data Preparation**: Loaded and cleaned the dataset, dropping non-essential columns and splitting data into features (X) and target (y) variables.
2. **Exploratory Data Analysis (EDA)**: Performed EDA to visualize trends and better understand the data.
3. **DNN Architecture for Regression**: Created a DNN model using PyTorch to predict continuous variables related to stock prices.
4. **Hyperparameter Tuning**: Employed `GridSearchCV` from sklearn to find optimal parameters (e.g., learning rate, optimizer, epochs, architecture).
5. **Model Training Visualization**: Plotted Loss vs. Epochs and Accuracy vs. Epochs for both training and test data, analyzing model convergence and areas of improvement.
6. **Regularization Techniques**: Implemented dropout, L1, and L2 regularization to reduce overfitting. Compared the results with the original model.

### Part 2: Multi-Class Classification for Predictive Maintenance

1. **Data Preprocessing**: Cleaned and standardized the dataset, handling missing values and normalizing feature scales.
2. **EDA and Data Augmentation**: Performed EDA and applied data augmentation techniques to balance the dataset.
3. **DNN Architecture for Classification**: Built a DNN model using PyTorch to classify maintenance types.
4. **Hyperparameter Tuning**: Used GridSearch to select the best hyperparameters for optimized performance.
5. **Model Evaluation and Visualization**: Visualized Loss and Accuracy across epochs, interpreting model accuracy and performance.
6. **Model Metrics Calculation**: Computed accuracy, sensitivity, and F1 score on both training and test datasets.
7. **Regularization Techniques**: Added dropout and other regularization methods to improve generalization, comparing performance with the initial model.

## Results

The lab demonstrated the capabilities of DNNs for both regression and classification tasks and highlighted the role of data preprocessing, hyperparameter tuning, and regularization in building effective models.

## Requirements

- Python 3.x
- PyTorch
- Scikit-Learn
- Pandas
- Matplotlib
- Seaborn

## Usage

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/deep-learning-lab.git
    ```
2. Install the required libraries:

3. Run the notebooks in Google Colab, Kaggle, or Jupyter Notebook.

## Notebooks Summary

- **NYSE Data Analysis and Regression**: The notebook `nyse-regression.ipynb` explores stock price data and implements a regression DNN model.
- **Predictive Maintenance Classification**: The notebook `predictive-maintenance-classification.ipynb` explores the predictive maintenance dataset and implements a multi-class classification model.

## Synthesis: Key Learnings from the Lab

This lab provided hands-on experience with PyTorch, focusing on deep learning architectures for regression and classification tasks. Through this project, we learned:

- **Data Preparation and Analysis**: The importance of thorough exploratory data analysis (EDA) and preprocessing techniques for understanding data structure, addressing imbalances, and standardizing inputs to improve model performance.
- **Deep Neural Network Design**: How to construct and implement Deep Neural Networks (DNNs) and Multi-Layer Perceptrons (MLPs) tailored for regression and classification problems, and the impact of various layers and architectures on performance.
- **Hyperparameter Tuning**: Utilizing GridSearch to optimize parameters (e.g., learning rate, epochs, optimizer choice), and understanding how these adjustments directly influence training efficiency and model accuracy.
- **Regularization Techniques**: Implementing dropout, L1, and L2 regularization methods to enhance model generalization, and observing how these techniques help reduce overfitting.
- **Model Evaluation and Metrics Interpretation**: The value of visualizing training/validation loss and accuracy over epochs, and calculating metrics such as accuracy, F1 score, and sensitivity to assess model quality on both training and test sets.
- **Data Augmentation**: Applying data augmentation to balance datasets, especially in multi-class classification tasks, and recognizing its impact on model robustness and generalization.

These key concepts, learned throughout the lab, are foundational for building effective deep learning models and optimizing them for real-world applications.

---



## Acknowledgments

Special thanks to Pr. Elaachak Lotfi at Université Abdelmalek Essaadi for guidance in the lab project.
