# Student Progress Tracking & Performance Prediction

A machine learning-based student performance tracking application that analyzes historical academic scores and predicts the next assessment performance using Random Forest Regression.

## Features
- Student information management
- Dynamic subject-wise performance input
- Prediction using a trained Random Forest Regression model
- Performance categorization
- Progress trend analysis
- Subject-wise visualizations
- Personalized academic recommendations

## Technology Stack
- Python
- Pandas and NumPy
- Scikit-learn
- Random Forest Regression
- Streamlit
- Joblib
- Google Colab

##Dataset

This project uses the Open University Learning Analytics Dataset (OULAD) for training and evaluating the student performance prediction model.

Dataset source: Open University Learning Analytics Dataset (OULAD)
https://analyse.kmi.open.ac.uk/open_dataset
The dataset contains student demographics, assessment records, course information and virtual learning environment activity.

Setup

Download the dataset from the official source.

Extract the CSV files.

Place the required files in the appropriate project data directory.

Run the preprocessing and model training notebook.

Note: The original dataset is not included in this GitHub repository. Please download it from the official source.

## Model Evaluation
- MAE: 9.31
- RMSE: 12.88
- R²: 0.3223

The model predicts the next assessment score using four previous performance scores and derived performance features.

## Limitations
The model is trained on OULAD assessment data. User-entered performance periods are used as inputs to the trained model and do not represent standardized monthly marks. Attendance and study hours are currently not used as model features.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
streamlit run app.py
```
