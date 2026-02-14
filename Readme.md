🩺 Diabetes Prediction using Neural Networks (Pima Indians Dataset)

A simple deep learning project that predicts whether a patient has diabetes using the Pima Indians Diabetes Dataset and a Feedforward Neural Network built with TensorFlow/Keras.

📌 Project Overview

This project builds a binary classification model to predict diabetes outcomes based on medical attributes such as:

Pregnancies

Glucose Level

Blood Pressure

Skin Thickness

Insulin

BMI

Diabetes Pedigree Function

Age

The dataset used is the Pima Indians Diabetes Database, originally from the UCI Machine Learning Repository.

📊 Dataset Information

Total Samples: 768

Features: 8 medical attributes

Target Variable: Outcome

0 → No Diabetes

1 → Diabetes

Dataset Source:

https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

🧠 Model Architecture

We implemented a simple Feedforward Neural Network using Keras:

Input Layer (8 features)

Dense Layer (4 neurons, sigmoid activation)

Dense Layer (2 neurons, tanh activation)

Output Layer (1 neuron, sigmoid activation)

🔧 Compilation

Optimizer: Adam

Loss Function: Binary Crossentropy

Metric: Accuracy

⚙️ Technologies Used

Python

Pandas

NumPy

Scikit-learn

TensorFlow / Keras

🚀 How to Run
1️⃣ Clone the Repository
git clone https://github.com/your-username/diabetes-prediction-nn.git
cd diabetes-prediction-nn

2️⃣ Install Dependencies
pip install pandas numpy scikit-learn tensorflow

3️⃣ Run the Script

You can run it in:

Google Colab

Jupyter Notebook

Local Python environment

📈 Training Details

Train/Test Split: 80/20

Validation Split: 10% (from training set)

Epochs: 100

Batch Size: 32

🎯 Model Performance

After training for 100 epochs:

Test Accuracy: ~74%

loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)


Output:

Test Accuracy: 0.7402

🧪 Project Workflow

Load Dataset

Data Preprocessing

Feature scaling using StandardScaler

Train-Test Split

Model Building

Model Training

Model Evaluation

📌 Key Learning Points

Binary classification using Neural Networks

Importance of feature scaling

Using validation split to monitor training

Understanding overfitting and model performance

🔮 Future Improvements

Add EarlyStopping

Hyperparameter tuning

Compare with Logistic Regression & Random Forest

Add Confusion Matrix & ROC Curve

Deploy as a simple web app (Flask / Streamlit)

📜 License

This project is for educational purposes.

👨‍💻 Author

Rabin Patel
