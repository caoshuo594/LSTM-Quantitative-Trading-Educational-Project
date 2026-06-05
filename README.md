# LSTM Quantitative Trading Educational Project - EURUSD H1 Strategy

An open-source educational project demonstrating how to build, train, validate, export, and deploy LSTM-based financial forecasting models using real EURUSD market data, PyTorch, ONNX, and MetaTrader 5.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-red)
![ONNX](https://img.shields.io/badge/ONNX-Supported-orange)
![MT5](https://img.shields.io/badge/MetaTrader5-Compatible-green)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

## Overview

This project provides a complete end-to-end quantitative trading workflow based on a Long Short-Term Memory (LSTM) neural network. Using real EURUSD historical market data, the model is trained in PyTorch, exported to ONNX format, and deployed directly in MetaTrader 5 (MT5) for backtesting and live inference.

The repository is designed as an educational resource for developers, students, quantitative traders, and machine learning practitioners who want to learn how deep learning can be applied to financial time-series forecasting and algorithmic trading.

The project covers the entire pipeline:

* Historical data acquisition from MetaTrader 5
* Data preprocessing and normalization
* LSTM model training with GPU acceleration
* Early stopping and validation monitoring
* ONNX model export
* Deployment inside MetaTrader 5 Expert Advisors (EA)
* Backtesting and strategy evaluation

## Key Features

### Deep Learning Based Forecasting

* LSTM neural network for time-series prediction
* Automatic feature extraction from OHLCV market data
* Support for CUDA GPU acceleration
* Validation-based model selection
* Early stopping to reduce overfitting

### Production Deployment

* ONNX model export for platform-independent deployment
* Native integration with MetaTrader 5
* Real-time inference inside MQL5 Expert Advisors
* Lightweight model size (~10 KB)

### Educational Focus

* Clear and well-documented code
* Complete training-to-deployment workflow
* Practical example using real EURUSD data
* Suitable for beginners in AI-powered quantitative trading

## Technology Stack

| Component               | Technology                   |
| ----------------------- | ---------------------------- |
| Deep Learning Framework | PyTorch 2.5+                 |
| Model Architecture      | LSTM                         |
| Model Format            | ONNX                         |
| Trading Platform        | MetaTrader 5                 |
| Languages               | Python 3.11, MQL5            |
| Data Source             | MetaTrader 5 Historical Data |

## Why LSTM for Quantitative Trading?

Long Short-Term Memory (LSTM) networks are specifically designed for sequential data and financial time-series forecasting.

Advantages include:

* Capturing long-term market dependencies
* Learning nonlinear price dynamics
* Automatic extraction of temporal patterns
* Improved gradient stability compared with traditional RNNs
* Strong performance on financial forecasting tasks

## Model Architecture

Input Shape:
(10 timesteps × 5 features)

Features:

* Open
* High
* Low
* Close
* Volume

Network Structure:

Input Layer
→ LSTM Layer (hidden_size=20)
→ Fully Connected Layer
→ Next-Candle Close Price Prediction

Model Statistics:

* Parameters: ~2,040
* ONNX Size: ~10 KB
* Inference Latency: <1 ms

## Training Pipeline

1. Retrieve EURUSD H1 historical data from MT5
2. Normalize OHLCV features using MinMaxScaler
3. Split data into:

   * 80% Training Set
   * 20% Validation Set
4. Train LSTM model using Adam optimizer
5. Monitor validation loss
6. Apply Early Stopping
7. Save best-performing model
8. Export ONNX model
9. Generate training loss visualization

## Trading Logic

For each newly completed H1 candle:

1. Retrieve the latest 10 historical candles
2. Normalize input features
3. Run ONNX inference
4. Convert prediction back to real price values
5. Generate trading signals:

Buy:
Predicted Price > Current Price × 1.0001

Sell:
Predicted Price < Current Price × 0.9999

Otherwise:
No Trade

## Educational Purpose

This repository is intended for:

* Machine Learning students
* Quantitative Trading beginners
* Forex traders interested in AI
* MetaTrader 5 developers
* Researchers studying financial forecasting

The project demonstrates how modern deep learning models can be integrated into traditional trading platforms and serves as a practical learning resource for AI-powered algorithmic trading.

## Disclaimer

This project is provided strictly for educational and research purposes.

It does not constitute financial advice, investment recommendations, or guarantees of future performance.

Financial markets are inherently uncertain, and historical performance does not guarantee future results. Any live trading conducted using this code is done entirely at the user's own risk.

## License

MIT License

Copyright (c) 2025 Cao Shuo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
