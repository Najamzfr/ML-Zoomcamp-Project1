from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Define the path to the models directory
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

# Load the model and DictVectorizer
model = joblib.load(os.path.join(models_dir, 'model.pkl'))
dv = joblib.load(os.path.join(models_dir, 'dv.pkl'))

def predict_loan(model, dv, df):
    X = dv.transform(df.to_dict(orient='records'))
    y_pred = model.predict(X)
    return y_pred[0]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        y_pred = predict_loan(model, dv, df)
        return jsonify({'prediction': float(y_pred)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)