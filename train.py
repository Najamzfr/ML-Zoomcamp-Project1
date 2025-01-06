from src.data import load_and_preprocess_data
from src.model import train_model

df_train, df_val, df_test, y_train, y_val, y_test = load_and_preprocess_data('data/loan_data.csv')
train_model(df_train, y_train)