# Neural Network from Scratch: Diabetes Prediction

## Overview
This project implements a **neural network from scratch** in Python using only **NumPy** for predicting diabetes based on health indicators. It also includes exploratory data analysis (EDA) and visualization of important features such as BMI, smoking habits, cholesterol levels, and healthcare access.

The neural network is a **2-layer feedforward network** with:
- Input layer
- Hidden layer (ReLU activation)
- Output layer (Sigmoid activation for binary classification)

---

## Dataset
The dataset used is the [Diabetes Binary 5050split Health Indicators BRFSS 2015](https://www.kaggle.com/datasets/fedesoriano/diabetes-health-indicators-dataset) CSV file.  

**Target Variable:**
- `Diabetes_binary`: 1 = Diabetes, 0 = No Diabetes

**Selected Features:**
- Age, BMI, MentHlth, PhysHlth, Cholesterol, Blood Pressure, Smoking, Alcohol consumption, Physical Activity, Healthcare access, Education, Income, Sex, Stroke, Heart Disease.

---

## Exploratory Data Analysis (EDA)
The project includes detailed analysis and visualization of:

- **BMI Distribution:** Identifying overweight and obese populations.
- **Smoking Effects:** Impact of smoking on heart disease and diabetes.
- **Blood Pressure & Stroke:** Stroke rates based on high blood pressure.
- **Cholesterol & Diabetes:** Effect of cholesterol and frequency of cholesterol checks.
- **Healthcare Access:** Impact on diabetes prevalence.
- **Cost Barriers:** Effect of doctor visit costs on diabetes.
- **Demographics:** Age, sex, education, income distributions.
- **Correlation Heatmaps:** For feature relationships.

Visualizations are generated using **Matplotlib** and **Seaborn**.

---

## Data Preprocessing
- Normalization using `MinMaxScaler` for continuous features (`Age`, `BMI`, `MentHlth`, `Income`).
- Train/Validation/Test split:
  - 70% Training
  - 20% Validation
  - 10% Test
- Creation of mini-batches for stochastic gradient descent.

---

## Neural Network Architecture
- **Input Layer:** Number of features
- **Hidden Layer:** 10 neurons, ReLU activation
- **Output Layer:** 1 neuron, Sigmoid activation (binary classification)
- **Loss Function:** Binary cross-entropy
- **Optimizer:** Gradient descent
- **Training:** Mini-batch stochastic gradient descent

The network is implemented fully from scratch with **forward propagation**, **backpropagation**, and **weight updates**.

---

## Training and Evaluation
- Training conducted for 100 epochs with batch size 32 and learning rate 0.01.
- Monitored metrics:
  - Training Error & Accuracy
  - Validation Error & Accuracy
- Final evaluation on the test set includes:
  - **Test Error**
  - **Test Accuracy**
  - **Confusion Matrix**

Plots include:
- Training vs Validation Error
- Training vs Validation Accuracy
- Confusion Matrix for Test Set

---

## Dependencies
- Python >= 3.8
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn

Install dependencies using:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn

Usage

Clone the repository:

git clone https://github.com/omarwaelll/neural-network-from-scratch.git
cd neural-network-from-scratch


Place the dataset CSV in the project folder.

Run the main script:

python neural_network_diabetes.py

