"""
Script to create Jupyter notebook for weather analysis
Run this to generate the notebook file
"""

import json
import os

print("Creating weather analysis notebook...")

# Ensure notebooks directory exists
os.makedirs('notebooks', exist_ok=True)

# Notebook content structure
notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Weather Prediction Data Analysis Notebook\n",
                "## Complete ML Portfolio Project\n",
                "\n",
                "This notebook demonstrates a complete machine learning pipeline for weather prediction."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Import required libraries\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "%matplotlib inline\n",
                "\n",
                "print('Libraries imported successfully')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Load Weather Dataset"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load the weather data\n",
                "df = pd.read_csv('../data/weather_data.csv', parse_dates=['date'])\n",
                "\n",
                "print(f'Dataset dimensions: {df.shape}')\n",
                "print(f'Date range: {df[\"date\"].min()} to {df[\"date\"].max()}')\n",
                "print('\\nFirst 5 rows of data:')\n",
                "df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Data Exploration and Visualization"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Plot temperature over time\n",
                "plt.figure(figsize=(12, 5))\n",
                "plt.plot(df['date'], df['temperature'], linewidth=1, alpha=0.7, color='blue')\n",
                "plt.xlabel('Date')\n",
                "plt.ylabel('Temperature (C)')\n",
                "plt.title('Temperature Trends Over Time')\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.xticks(rotation=45)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Temperature distribution histogram\n",
                "plt.figure(figsize=(10, 5))\n",
                "plt.hist(df['temperature'], bins=30, edgecolor='black', alpha=0.7, color='orange')\n",
                "plt.xlabel('Temperature (C)')\n",
                "plt.ylabel('Frequency')\n",
                "plt.title('Temperature Distribution')\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Feature Engineering"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.ensemble import RandomForestRegressor\n",
                "from sklearn.metrics import mean_absolute_error\n",
                "\n",
                "# Create time-based features\n",
                "df['temp_yesterday'] = df['temperature'].shift(1)\n",
                "df['temp_week_ago'] = df['temperature'].shift(7)\n",
                "df['target_temperature'] = df['temperature'].shift(-1)\n",
                "df = df.dropna()\n",
                "\n",
                "# Prepare feature matrix and target vector\n",
                "X = df[['temperature', 'humidity', 'temp_yesterday', 'temp_week_ago']]\n",
                "y = df['target_temperature']\n",
                "\n",
                "# Split into training and testing sets\n",
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
                "\n",
                "print(f'Training samples: {len(X_train)}')\n",
                "print(f'Testing samples: {len(X_test)}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Machine Learning Model"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Initialize and train Random Forest model\n",
                "model = RandomForestRegressor(n_estimators=100, random_state=42)\n",
                "model.fit(X_train, y_train)\n",
                "\n",
                "# Make predictions on test set\n",
                "y_pred = model.predict(X_test)\n",
                "\n",
                "# Calculate model accuracy\n",
                "mae = mean_absolute_error(y_test, y_pred)\n",
                "print(f'Model Mean Absolute Error: {mae:.2f} degrees C')\n",
                "print('Lower values indicate better accuracy')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Model Evaluation"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Visualize predictions vs actual values\n",
                "plt.figure(figsize=(10, 6))\n",
                "plt.scatter(y_test, y_pred, alpha=0.6, s=30, color='green')\n",
                "plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)\n",
                "plt.xlabel('Actual Temperature (C)')\n",
                "plt.ylabel('Predicted Temperature (C)')\n",
                "plt.title('Model Performance: Actual vs Predicted')\n",
                "plt.grid(True, alpha=0.3)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Generate Forecast"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Simple forecasting function\n",
                "def generate_forecast(model, current_features, days=7):\n",
                "    forecasts = []\n",
                "    features = current_features.copy()\n",
                "    \n",
                "    for day in range(days):\n",
                "        pred_temp = model.predict([features])[0]\n",
                "        forecasts.append(pred_temp)\n",
                "        # Update features for next prediction\n",
                "        features[0] = pred_temp\n",
                "    \n",
                "    return forecasts\n",
                "\n",
                "# Get latest data for forecasting\n",
                "latest_features = X.iloc[-1].values\n",
                "forecast_temperatures = generate_forecast(model, latest_features, days=7)\n",
                "\n",
                "# Display forecast\n",
                "print('7-Day Temperature Forecast:')\n",
                "print('-' * 30)\n",
                "for i, temp in enumerate(forecast_temperatures, 1):\n",
                "    print(f'Day {i}: {temp:.1f} C')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## Project Summary"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print('='*60)\n",
                "print('PROJECT SUMMARY')\n",
                "print('='*60)\n",
                "print(f'\\nDataset size: {len(df)} days of weather data')\n",
                "print(f'ML Algorithm: Random Forest Regressor')\n",
                "print(f'Model Accuracy: {mae:.2f} degrees C Mean Absolute Error')\n",
                "print(f'Forecast Generated: 7-day temperature prediction')\n",
                "print('\\nThis project demonstrates:')\n",
                "print('- Data preprocessing and feature engineering')\n",
                "print('- Machine learning model training and evaluation')\n",
                "print('- Time series forecasting')\n",
                "print('- Data visualization and analysis')\n",
                "print('='*60)\n",
                "print('\\nComplete ML portfolio project ready!')"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Save notebook to file
with open('notebooks/weather_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook_content, f, indent=2)

print("Notebook created: notebooks/weather_analysis.ipynb")
print("Open the file in VS Code to run the notebook cells")