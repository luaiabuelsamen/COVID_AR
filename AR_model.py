from flask import Flask, render_template, jsonify
import yfinance as yf
import numpy as np
from scipy import linalg
import pandas as pd
from datetime import datetime, timedelta

app = Flask(__name__)

def form_A_b(y, ell):
    N = y.size
    A = np.ones((N - ell, ell))
    for n in range(N - ell):
        A[n] = y[n : ell + n]
    b = y[ell : N]
    return A, b

def fit(A, b):
    return linalg.solve(A.T @ A, A.T @ b)

def predict(y, beta):
    N = y.size
    ell = beta.size
    y_pred = np.zeros_like(y)
    for n in range(ell, N):
        y_pred[n] = beta.reshape(1, -1) @ y[n - ell : n].reshape(-1, 1)
    # Set the first ell predictions to the average of the first ell measurements
    y_pred_mean = np.mean(y[:ell])
    y_pred[:ell] = np.ones(ell) * y_pred_mean
    return y_pred

def get_stock_data(symbol, period='1y'):
    stock = yf.Ticker(symbol)
    df = stock.history(period=period)
    return df['Close'].values

def train_model(data, ell=16):
    y_scale = np.max(data)
    y_normalized = data / y_scale
    
    A, b = form_A_b(y_normalized, ell)
    beta = fit(A, b)
    
    return beta, y_scale

def make_prediction(data, beta, y_scale, future_days=30):
    # Normalize the data
    y_normalized = data / y_scale
    
    # Make prediction for existing data
    y_pred = predict(y_normalized, beta)
    
    # Scale predictions back
    y_pred = y_pred * y_scale
    
    # Generate future predictions
    future_pred = []
    last_values = data[-beta.size:]
    
    for _ in range(future_days):
        next_pred = (beta.reshape(1, -1) @ (last_values[-beta.size:] / y_scale).reshape(-1, 1))[0] * y_scale
        future_pred.append(float(next_pred))
        last_values = np.append(last_values[1:], next_pred)
    
    return y_pred, future_pred

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict/<symbol>')
def predict_stock(symbol):
    try:
        # Get historical data
        data = get_stock_data(symbol)
        
        # Train model
        beta, y_scale = train_model(data)
        
        # Make predictions
        historical_pred, future_pred = make_prediction(data, beta, y_scale)
        
        # Prepare dates
        dates = [(datetime.now() - timedelta(days=len(data)-i)).strftime('%Y-%m-%d') 
                for i in range(len(data))]
        future_dates = [(datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d') 
                       for i in range(1, len(future_pred)+1)]
        
        return jsonify({
            'success': True,
            'historical_dates': dates,
            'historical_actual': data.tolist(),
            'historical_predicted': historical_pred.tolist(),
            'future_dates': future_dates,
            'future_predicted': future_pred
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)