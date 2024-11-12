from flask import Flask, render_template, jsonify, request
import yfinance as yf
import numpy as np
from scipy import linalg
import pandas as pd
from datetime import datetime, timedelta
from flask_socketio import SocketIO, emit
import threading
import time
import talib

app = Flask(__name__)
socketio = SocketIO(app)


def form_A_b(y, ell):
    """
    Form the A matrix and b vector for AR model fitting.
    
    Args:
        y (numpy.array): Input time series data
        ell (int): AR model order
        
    Returns:
        tuple: (A matrix, b vector) for solving the AR model coefficients
    """
    N = y.size
    A = np.ones((N - ell, ell))
    for n in range(N - ell):
        A[n] = y[n : ell + n]
    b = y[ell : N]
    return A, b

def fit(A, b):
    """
    Fit the AR model using least squares.
    
    Args:
        A (numpy.array): A matrix from form_A_b
        b (numpy.array): b vector from form_A_b
        
    Returns:
        numpy.array: AR model coefficients
    """
    # Use ridge regression with small alpha to improve stability
    alpha = 1e-6
    I = np.eye(A.shape[1])
    return linalg.solve(A.T @ A + alpha * I, A.T @ b)

def train_model(data, ell=16):
    """
    Train an enhanced AR model with adaptive order selection.
    
    Args:
        data (numpy.array): Input time series data
        ell (int, optional): Initial AR model order. Defaults to 16.
        
    Returns:
        tuple: (model coefficients, scale factor, model order)
    """
    # Scale the data to prevent numerical issues
    y_scale = np.max(np.abs(data))
    y_normalized = data / y_scale
    
    # Try different model orders and select the best one using AIC
    best_aic = np.inf
    best_beta = None
    best_ell = ell
    
    # Test different model orders around the initial guess
    for test_ell in range(max(4, ell-4), min(ell+5, len(data)//4)):
        A, b = form_A_b(y_normalized, test_ell)
        beta = fit(A, b)
        
        # Calculate predictions for AIC
        y_pred = predict(y_normalized, beta)
        
        # Calculate AIC
        n = len(data)
        mse = np.mean((y_normalized - y_pred) ** 2)
        aic = n * np.log(mse) + 2 * test_ell  # AIC formula
        
        if aic < best_aic:
            best_aic = aic
            best_beta = beta
            best_ell = test_ell
    
    return best_beta, y_scale, best_ell

def predict(y, beta):
    """
    Make predictions using the AR model.
    
    Args:
        y (numpy.array): Input time series data
        beta (numpy.array): AR model coefficients
        
    Returns:
        numpy.array: Predicted values
    """
    N = y.size
    ell = beta.size
    y_pred = np.zeros_like(y)
    
    # Set the first ell predictions to the actual values
    y_pred[:ell] = y[:ell]
    
    # Make predictions for the rest
    for n in range(ell, N):
        y_pred[n] = beta.reshape(1, -1) @ y[n - ell : n].reshape(-1, 1)
    
    return y_pred

def make_prediction(data, beta, y_scale, future_days=30):
    """
    Generate historical and future predictions with confidence intervals.
    
    Args:
        data (numpy.array): Historical data
        beta (numpy.array): AR model coefficients
        y_scale (float): Scale factor used during training
        future_days (int, optional): Number of days to predict. Defaults to 30.
        
    Returns:
        tuple: (historical predictions, future predictions, confidence intervals)
    """
    # Normalize the data
    y_normalized = data / y_scale
    
    # Make prediction for existing data
    y_pred = predict(y_normalized, beta)
    historical_pred = y_pred * y_scale
    
    # Generate future predictions with confidence intervals
    future_pred = []
    conf_intervals = []
    last_values = data[-beta.size:] / y_scale
    
    # Calculate prediction error variance from historical predictions
    pred_error_var = np.var(y_normalized - y_pred)
    
    for i in range(future_days):
        # Make next prediction
        next_pred = (beta.reshape(1, -1) @ last_values[-beta.size:].reshape(-1, 1))[0]
        
        # Scale prediction back
        scaled_pred = float(next_pred * y_scale)
        future_pred.append(scaled_pred)
        
        # Calculate confidence interval (95%)
        conf_width = 1.96 * np.sqrt(pred_error_var * (i + 1)) * y_scale
        conf_intervals.append([scaled_pred - conf_width, scaled_pred + conf_width])
        
        # Update last values for next prediction
        last_values = np.append(last_values[1:], next_pred)
    
    return historical_pred, future_pred, conf_intervals

def calculate_model_diagnostics(data, predictions):
    """
    Calculate various model diagnostic metrics.
    
    Args:
        data (numpy.array): Actual historical data
        predictions (numpy.array): Model predictions
        
    Returns:
        dict: Dictionary containing diagnostic metrics
    """
    # Calculate basic error metrics
    errors = data - predictions
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(errors / data)) * 100
    
    # Calculate R-squared
    ss_tot = np.sum((data - np.mean(data)) ** 2)
    ss_res = np.sum(errors ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Calculate Durbin-Watson statistic for autocorrelation
    error_diff = np.diff(errors)
    dw_stat = np.sum(error_diff ** 2) / np.sum(errors ** 2)
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape),
        'r_squared': float(r_squared),
        'durbin_watson': float(dw_stat)
    }

def get_technical_indicators(data):
    close = data['Close'].values
    high = data['High'].values
    low = data['Low'].values
    volume = data['Volume'].values
    
    # Calculate various technical indicators
    indicators = {
        'rsi': talib.RSI(close),
        'macd': talib.MACD(close)[0],
        'bbands': talib.BBANDS(close),
        'atr': talib.ATR(high, low, close),
        'obv': talib.OBV(close, volume)
    }
    
    return indicators

def enhance_prediction(data, indicators):
    # Combine traditional AR prediction with technical indicators
    # This is a simplified example - you could make this more sophisticated
    rsi = indicators['rsi']
    macd = indicators['macd']
    
    # Adjust predictions based on RSI and MACD
    prediction_adjustment = np.zeros_like(data)
    
    # RSI adjustment
    rsi_factor = 0.001  # Adjustment strength
    prediction_adjustment += (rsi - 50) * rsi_factor
    
    # MACD adjustment
    macd_factor = 0.01  # Adjustment strength
    prediction_adjustment += macd * macd_factor
    
    return data * (1 + prediction_adjustment)

def calculate_confidence_metrics(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    mae = np.mean(np.abs(actual - predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'mape': float(mape)
    }

@app.route('/predict/<symbol>')
def predict_stock(symbol):
    try:
        timeframe = request.args.get('timeframe', '1y')
        prediction_days = int(request.args.get('prediction_days', 30))
        
        # Get historical data
        stock = yf.Ticker(symbol)
        df = stock.history(period=timeframe)
        
        if df.empty:
            raise ValueError(f"No data available for symbol {symbol}")
        
        # Calculate technical indicators
        indicators = get_technical_indicators(df)
        
        # Get closing prices and prepare for modeling
        data = df['Close'].values
        
        # Train the enhanced AR model
        beta, y_scale, model_order = train_model(data)
        
        # Generate predictions with confidence intervals
        historical_pred, future_pred, conf_intervals = make_prediction(
            data, beta, y_scale, prediction_days
        )
        
        # Calculate model diagnostics
        diagnostics = calculate_model_diagnostics(data, historical_pred)
        
        # Prepare dates
        dates = df.index.strftime('%Y-%m-%d').tolist()
        future_dates = [
            (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
            for i in range(1, prediction_days + 1)
        ]
        
        return jsonify({
            'success': True,
            'historical_dates': dates,
            'historical_actual': data.tolist(),
            'historical_predicted': historical_pred.tolist(),
            'future_dates': future_dates,
            'future_predicted': future_pred,
            'confidence_intervals': conf_intervals,
            'model_diagnostics': diagnostics,
            'technical_indicators': {
                'rsi': indicators['rsi'].tolist(),
                'macd': indicators['macd'].tolist(),
                'volume': df['Volume'].tolist()
            },
            'model_info': {
                'order': int(model_order),
                'scale_factor': float(y_scale)
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def real_time_updates():
    """Background task to simulate real-time price updates"""
    while True:
        for symbol in active_symbols:
            try:
                stock = yf.Ticker(symbol)
                price = stock.info['regularMarketPrice']
                socketio.emit(f'price_update_{symbol}', {'price': price})
            except:
                continue
        time.sleep(5)  # Update every 5 seconds

active_symbols = set()

@socketio.on('connect')
def handle_connect():
    if not hasattr(app, 'update_thread'):
        app.update_thread = threading.Thread(target=real_time_updates)
        app.update_thread.daemon = True
        app.update_thread.start()

@socketio.on('subscribe')
def handle_subscribe(symbol):
    active_symbols.add(symbol)

@socketio.on('unsubscribe')
def handle_unsubscribe(symbol):
    active_symbols.remove(symbol)

if __name__ == '__main__':
    socketio.run(app, debug=True)