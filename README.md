# Graduate Admission Prediction using Neural Network

## Overview

The **Graduate Admission Prediction** project aims to predict the chances of a student getting admitted to a graduate program based on various factors such as GRE scores, TOEFL scores, GPA, research experience, etc. A neural network is used to model and predict the admission chances, leveraging the power of deep learning to achieve higher accuracy.

## Problem Statement

With increasing competition in graduate admissions, students want to know their chances of getting accepted into a graduate program. This project provides a solution by building a predictive model that estimates the probability of admission based on key admission criteria.

## Features

- **Data Preprocessing**: Cleans and scales the input data for better neural network performance.
- **Neural Network Model**: Implements a fully connected feed-forward neural network using frameworks like TensorFlow or PyTorch.
- **Performance Metrics**: Evaluates model accuracy, precision, recall, and F1-score.
- **Prediction Output**: Provides a probability score (0 to 1) indicating the likelihood of admission.

## Dataset

The dataset contains information on the following features:

- **GRE Score**: Graduate Record Exam score (out of 340)
- **TOEFL Score**: Test of English as a Foreign Language score (out of 120)
- **University Rating**: Rating of the university (1 to 5)
- **LOR**: Letter of Recommendation strength (out of 5)
- **CGPA**: Undergraduate GPA (on a 10-point scale)
- **Research**: Whether the student has research experience (0 or 1)
- **Chance of Admit**: The target variable, representing the likelihood of admission (0 to 1)

## Neural Network Architecture

- **Input Layer**: Accepts the scaled input features.
- **Hidden Layers**: Two hidden layers with ReLU activation.
- **Output Layer**: One neuron with a sigmoid activation function to predict the admission probability.
- **Loss Function**: Binary Cross-Entropy.
- **Optimizer**: Adam Optimizer for weight updates.

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - **TensorFlow/Keras**: For building and training the neural network
  - **Pandas**: For data manipulation
  - **NumPy**: For numerical computations
  - **Matplotlib/Seaborn**: For data visualization
  - **Scikit-learn**: For data preprocessing and model evaluation

