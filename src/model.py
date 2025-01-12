import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

def train_model(df_train, y_train):
    train_dict = df_train.to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(train_dict)
    
    model = LogisticRegression(solver='lbfgs', max_iter=1000, C=0.01, random_state=42)
    model.fit(X_train, y_train)
    
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(dv, 'models/dv.pkl')
    
def predict(model, dv, df):
    dicts = df.to_dict(orient='records')
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]
    return y_pred