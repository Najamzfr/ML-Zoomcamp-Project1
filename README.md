# Loan Approval Classification

This project is a machine learning-based loan approval classification system. It uses a logistic regression model to predict whether a loan application should be approved or not based on various features such as age, employment experience, home ownership, loan intent, loan grade, loan amount, and loan interest rate.

The project is deployed as a REST API using Flask, allowing users to send loan application data and receive a prediction.

The Dataset used in this project is obtained from  `kaggle` click the link  ->  [Data Source](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)

---

## Project Structure
```bash
loan_prediction/
├── data/
│   └── loan_data.csv               # Dataset used for training and testing
├── models/
│   ├── model.pkl                   # Trained logistic regression model
│   └── dv.pkl                      # DictVectorizer for feature transformation
├── src/
│   ├── data.py                     # Script for loading and preprocessing data
│   └── model.py                    # Script for training and saving the model
├── app/
│   ├── app.py                      # Flask application for serving predictions
│   └── requirements.txt            # Python dependencies for the Flask app
├── train.py                        # Script to train the model
├── run.py                          # Script to run the Flask app
└── README.md                       # Project documentation
```
---

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package manager)

---

## Setup Instructions

### 1. Clone the Repository


git clone https://github.com/your-username/loan-approval-classification.git
cd loan-approval-classification

### 2. Install Dependencies
Install the required Python packages:

```bash
pip install -r app/requirements.txt
```

### 3. Train the Model
Run the train.py script to train the model and save it to the models directory:

```bash
python train.py
```
This will generate two files in the models directory:

model.pkl: The trained logistic regression model.

dv.pkl: The DictVectorizer used for feature transformation.

Running the Flask App
To start the Flask application, run:

```bash
python run.py
```
The Flask app will start on http://127.0.0.1:5000.

API Usage
Endpoint: /predict
Method: POST

Description: Predicts whether a loan application should be approved or not.

Input: JSON object containing loan application data.

Output: JSON object containing the prediction.

Example Request
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{
    "person_age": 30,
    "person_emp_exp": 5,
    "person_home_ownership": "Rent",
    "loan_intent": "Personal",
    "loan_grade": "C",
    "loan_amnt": 10000,
    "loan_int_rate": 12.0
}'
```
Example Response
```bash

{
    "prediction": 1.0
}
```
`prediction`: `1.0` indicates the loan is approved, 0.0 indicates the loan is not approved.

Model Training Details
### Data Preprocessing
The dataset is cleaned by removing rows with missing values and filtering out unrealistic values (e.g., age > 90, employment experience > 50 years).

Features like person_education and person_gender are dropped during preprocessing.

### Model
A logistic regression model is trained using scikit-learn.

The model is saved as model.pkl, and the feature transformation pipeline is saved as dv.pkl.
