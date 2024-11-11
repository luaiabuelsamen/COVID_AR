# Stock Price Predictor

This is a web application that predicts the future stock prices of a given company using historical stock data. It leverages a simple autoregressive (AR) model to forecast stock prices, and presents the results through a visual chart. The app allows users to input a stock symbol and view both historical and predicted stock prices in an interactive line chart.

## Features

- **Stock Price Prediction**: The application fetches historical stock data from Yahoo Finance and uses an autoregressive model to predict future stock prices.
- **Interactive Chart**: Predictions are displayed in a line chart, showing both actual historical prices and predicted future prices.
- **User-Friendly Interface**: The user can simply enter a stock symbol and get the prediction in seconds.

## Requirements

- **Python 3.7+** (for backend)
- **Flask**: A lightweight web framework for Python.
- **yfinance**: A library for fetching financial data from Yahoo Finance.
- **SciPy**: Used for linear algebra operations.
- **pandas**: Used for data handling and manipulation.
- **Chart.js**: A JavaScript library for drawing charts.

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/stock-price-predictor.git
   ```

2. Navigate to the project directory:

   ```bash
   cd stock-price-predictor
   ```

3. Set up a virtual environment (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

4. Install the necessary Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:

   ```bash
   python app.py
   ```

6. Open your browser and navigate to `http://127.0.0.1:5000/` to access the app.

## Usage

1. Upon loading the app, enter a stock symbol (e.g., "AAPL" for Apple) into the input field and click **Get Prediction**.
2. The application will fetch historical data for the stock and display a line chart showing both actual stock prices and the model's predictions.
3. You can enter different stock symbols to view predictions for other stocks.

## How It Works

1. **Data Fetching**: The app uses the `yfinance` library to fetch historical stock data from Yahoo Finance.
2. **Model Training**: The historical data is normalized, and an autoregressive model is trained using the past data.
3. **Prediction**: The model makes predictions for the next 30 days based on the trained parameters, and the results are returned to the frontend.
4. **Visualization**: The predictions and actual prices are rendered in an interactive line chart using Chart.js.

## Folder Structure

```
.
├── app.py                 # Main Flask app file
├── templates/
│   └── index.html         # Frontend HTML for displaying the app
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Future Enhancements

- **Advanced Models**: Implement more advanced forecasting models like ARIMA or LSTM.
- **User Inputs**: Allow users to choose prediction periods or select different models.
- **Multiple Stock Symbols**: Support predictions for multiple stock symbols in a single chart.

## Contributing

Contributions are welcome! If you'd like to help improve the app, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.