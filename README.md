# Weather Prediction Machine Learning Project

A complete machine learning project for predicting weather temperatures using historical data and Random Forest regression.

## Project Overview

This project implements a complete machine learning pipeline for weather prediction, including data generation, feature engineering, model training, evaluation, and forecasting. The system predicts next-day temperatures with high accuracy and can generate multi-day forecasts.

## Features

- Synthetic weather data generation with realistic seasonal patterns
- Feature engineering including lag features, rolling statistics, and seasonal transforms
- Random Forest regression model for temperature prediction
- Comprehensive model evaluation with MAE, RMSE, and R-squared metrics
- 7-day weather forecast generation
- Interactive demonstration mode
- Professional data visualizations
- Jupyter notebook for data analysis

## Project Structure
weather-predictor/
├── data/ # Weather datasets

├── models/ # Saved ML models

├── notebooks/ # Jupyter notebooks

├── results/ # Output visualizations

├── .gitignore # Git ignore file

├── create_notebook.py # Script to create notebook

├── README.md # Project documentation

├── requirements.txt # Python dependencies

└── weather_predictor.py # Main application


## Installation

1. Ensure Python 3.8+ is installed
2. Install required packages:
   ```bash
   pip install -r requirements.txt
