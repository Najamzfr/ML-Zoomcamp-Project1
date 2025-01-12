import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(path):
    df = pd.read_csv(path)
    
    # Data cleaning
    df = df.dropna()
    df = df[(df['person_age'] < 90) & (df['person_emp_exp'] <= 50)]
    
    # Feature engineering
    df = df.drop(columns=['person_education', 'person_gender'])
    
    # Train test split
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
    
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    
    y_train = df_train['loan_status'].values
    y_val = df_val['loan_status'].values
    y_test = df_test['loan_status'].values
    
    del df_train['loan_status']
    del df_val['loan_status']
    del df_test['loan_status']
    
    return df_train, df_val, df_test, y_train, y_val, y_test