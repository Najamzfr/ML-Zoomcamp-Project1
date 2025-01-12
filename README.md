# Loan Approval Classification

This project is a machine learning-based loan approval classification system. It uses a logistic regression model to predict whether a loan application should be approved or not based on various features such as age, employment experience, home ownership, loan intent, loan grade, loan amount, and loan interest rate.

The project is deployed as a REST API using Flask, allowing users to send loan application data and receive a prediction.

---

## Project Structure
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

---

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package manager)

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/loan-approval-classification.git
cd loan-approval-classification

### 2. Install Dependencies
Install the required Python packages:

```bash
pip install -r app/requirements.txt

