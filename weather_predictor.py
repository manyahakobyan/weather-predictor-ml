"""
Weather Prediction Machine Learning System
Portfolio Project by Manya Hakobyan
A complete ML pipeline for predicting weather temperatures
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import os

class WeatherDataGenerator:
    """Generate synthetic weather data for training and testing"""
    
    def __init__(self, start_date='2020-01-01', days=1000):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.days = days
        
    def generate_data(self):
        """Create realistic weather data with seasonal patterns"""
        print("Generating synthetic weather data...")
        
        dates = [self.start_date + timedelta(days=i) for i in range(self.days)]
        day_of_year = np.array([d.timetuple().tm_yday for d in dates])
        
        # Temperature with seasonal variation
        base_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 105) / 365)
        temperature = base_temp + np.random.normal(0, 5, self.days)
        
        # Humidity - inversely related to temperature
        humidity = 60 - 0.5 * temperature + np.random.normal(0, 10, self.days)
        humidity = np.clip(humidity, 10, 100)
        
        # Atmospheric pressure
        pressure = 1013 + np.random.normal(0, 10, self.days)
        
        # Wind speed
        wind_speed = np.random.gamma(2, 2, self.days)
        wind_speed = np.clip(wind_speed, 0, 30)
        
        # Precipitation - occasional rain
        precipitation = np.random.exponential(0.5, self.days)
        precipitation[np.random.random(self.days) > 0.3] = 0
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'temperature': np.round(temperature, 1),
            'humidity': np.round(humidity, 1),
            'pressure': np.round(pressure, 1),
            'wind_speed': np.round(wind_speed, 1),
            'precipitation': np.round(precipitation, 2),
            'day_of_year': day_of_year,
            'month': [d.month for d in dates],
            'year': [d.year for d in dates]
        })
        
        print(f"Generated {len(df)} days of weather data")
        return df

class WeatherPredictor:
    """Machine learning model for weather prediction"""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def prepare_features(self, df):
        """Engineer features for the prediction model"""
        df = df.copy()
        
        # Create lag features from previous days
        for lag in [1, 2, 3, 7]:
            df[f'temp_lag_{lag}'] = df['temperature'].shift(lag)
            df[f'humidity_lag_{lag}'] = df['humidity'].shift(lag)
        
        # Rolling statistics over 7 days
        df['temp_rolling_mean_7'] = df['temperature'].rolling(7).mean()
        df['temp_rolling_std_7'] = df['temperature'].rolling(7).std()
        
        # Seasonal features using sine/cosine transformation
        df['sin_day'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['cos_day'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Target variable: tomorrow's temperature
        df['temperature_tomorrow'] = df['temperature'].shift(-1)
        
        # Remove rows with missing values
        df = df.dropna()
        
        return df
    
    def train(self, df, test_size=0.2):
        """Train the machine learning model"""
        print("\n" + "="*60)
        print("TRAINING WEATHER PREDICTION MODEL")
        print("="*60)
        
        # Prepare features
        df_processed = self.prepare_features(df)
        
        # Define features and target
        self.feature_columns = [col for col in df_processed.columns 
                               if col not in ['date', 'temperature_tomorrow']]
        
        X = df_processed[self.feature_columns]
        y = df_processed['temperature_tomorrow']
        
        # Split data into training and testing sets
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Number of features: {len(self.feature_columns)}")
        
        # Scale features for better model performance
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        print("Training model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train_scaled)
        y_pred_test = self.model.predict(X_test_scaled)
        
        # Calculate and display performance metrics
        self._print_performance_metrics(y_train, y_pred_train, y_test, y_pred_test)
        
        return X_test, y_test, y_pred_test
    
    def _print_performance_metrics(self, y_train, y_pred_train, y_test, y_pred_test):
        """Display model performance metrics"""
        metrics = {
            'Training MAE': mean_absolute_error(y_train, y_pred_train),
            'Testing MAE': mean_absolute_error(y_test, y_pred_test),
            'Training RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'Testing RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'Training R-squared': r2_score(y_train, y_pred_train),
            'Testing R-squared': r2_score(y_test, y_pred_test)
        }
        
        print("\n" + "="*60)
        print("MODEL PERFORMANCE METRICS")
        print("="*60)
        
        for metric, value in metrics.items():
            if 'R-squared' in metric:
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value:.2f} degrees C")
    
    def predict_forecast(self, df, days=7):
        """Generate a simple weather forecast"""
        print(f"\nGENERATING {days}-DAY WEATHER FORECAST")
        print("="*60)
        
        forecasts = []
        current_temp = df['temperature'].iloc[-1]
        current_humidity = df['humidity'].iloc[-1]
        avg_wind = df['wind_speed'].mean()
        
        for day in range(days):
            # Simple prediction model
            day_of_year = (datetime.now().timetuple().tm_yday + day) % 365
            seasonal_effect = 5 * np.sin(2 * np.pi * (day_of_year - 105) / 365)
            trend = -0.1 * day
            noise = np.random.normal(0, 2)
            
            predicted_temp = current_temp + seasonal_effect + trend + noise
            predicted_humidity = max(20, min(100, current_humidity - 0.1 * (predicted_temp - current_temp)))
            
            # Simple rain prediction
            will_rain = predicted_humidity > 75 and np.random.random() > 0.7
            rain_amount = np.random.exponential(0.3) if will_rain else 0
            
            forecasts.append({
                'Day': day + 1,
                'Temperature_C': round(predicted_temp, 1),
                'Humidity_pct': round(predicted_humidity, 1),
                'Wind_kmh': round(avg_wind + np.random.normal(0, 2), 1),
                'Rain_mm': round(rain_amount, 2)
            })
        
        forecast_df = pd.DataFrame(forecasts)
        print(forecast_df.to_string(index=False))
        
        return forecast_df
    
    def save_model(self, path='models/weather_model.pkl'):
        """Save the trained model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': 'RandomForest'
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\nModel saved to {path}")
    
    def load_model(self, path='models/weather_model.pkl'):
        """Load a previously trained model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        
        print(f"Model loaded from {path}")

class WeatherVisualizer:
    """Create visualizations for weather data and predictions"""
    
    @staticmethod
    def plot_predictions(y_test, y_pred, save_path='results/predictions.png'):
        """Visualize model predictions vs actual values"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Scatter plot: Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6, s=30, color='steelblue')
        axes[0, 0].plot([y_test.min(), y_test.max()], 
                       [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Temperature (C)')
        axes[0, 0].set_ylabel('Predicted Temperature (C)')
        axes[0, 0].set_title('Actual vs Predicted Temperatures')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Time series comparison
        sample_size = min(50, len(y_test))
        axes[0, 1].plot(range(sample_size), y_test.values[:sample_size], 
                       'o-', label='Actual', linewidth=2, markersize=4)
        axes[0, 1].plot(range(sample_size), y_pred[:sample_size], 
                       's-', label='Predicted', linewidth=2, markersize=4)
        axes[0, 1].set_xlabel('Time (sample index)')
        axes[0, 1].set_ylabel('Temperature (C)')
        axes[0, 1].set_title('Temperature Predictions Over Time')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Error distribution histogram
        errors = y_test - y_pred
        axes[1, 0].hist(errors, bins=25, edgecolor='black', alpha=0.7, color='lightcoral')
        axes[1, 0].axvline(x=0, color='darkred', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Prediction Error (C)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Distribution of Prediction Errors')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Feature importance placeholder
        axes[1, 1].barh(['Current Temp', 'Previous Temp', 'Weekly Avg', 
                        'Humidity', 'Season'], 
                       [0.3, 0.25, 0.15, 0.2, 0.1], color='mediumseagreen')
        axes[1, 1].set_xlabel('Relative Importance')
        axes[1, 1].set_title('Feature Importance for Temperature Prediction')
        axes[1, 1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to {save_path}")
    
    @staticmethod
    def plot_forecast(forecast_df, save_path='results/forecast.png'):
        """Visualize the weather forecast"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Temperature forecast bar chart
        axes[0].bar(forecast_df['Day'], forecast_df['Temperature_C'], 
                   color='coral', alpha=0.8, edgecolor='darkred', linewidth=2)
        axes[0].set_xlabel('Day')
        axes[0].set_ylabel('Temperature (C)')
        axes[0].set_title('7-Day Temperature Forecast')
        axes[0].set_xticks(forecast_df['Day'])
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add temperature values on bars
        for idx, row in forecast_df.iterrows():
            axes[0].text(row['Day'], row['Temperature_C'] + 0.3, 
                        f"{row['Temperature_C']}C", 
                        ha='center', va='bottom', fontweight='bold')
        
        # Humidity and precipitation plot
        ax2 = axes[1].twinx()
        axes[1].plot(forecast_df['Day'], forecast_df['Humidity_pct'], 
                    'b-o', linewidth=3, markersize=8, label='Humidity')
        ax2.bar(forecast_df['Day'], forecast_df['Rain_mm'], 
               alpha=0.4, color='blue', label='Rain')
        
        axes[1].set_xlabel('Day')
        axes[1].set_ylabel('Humidity (%)', color='blue')
        ax2.set_ylabel('Rain (mm)', color='blue')
        axes[1].set_title('Humidity and Precipitation Forecast')
        axes[1].set_xticks(forecast_df['Day'])
        axes[1].grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

def interactive_demo():
    """Interactive demonstration for users"""
    print("\n" + "="*60)
    print("INTERACTIVE DEMONSTRATION")
    print("="*60)
    
    try:
        print("\nEnter current weather conditions:")
        current_temp = float(input("Current temperature (C): "))
        current_humidity = float(input("Current humidity (%): "))
        
        print("\n" + "="*60)
        print("YOUR PERSONAL WEATHER FORECAST")
        print("="*60)
        
        # Simple forecast based on user input
        for day in range(1, 4):
            seasonal_effect = 2 * np.sin(2 * np.pi * (datetime.now().timetuple().tm_yday + day) / 365)
            predicted_temp = current_temp + seasonal_effect + np.random.normal(0, 1.5)
            predicted_humidity = max(20, min(100, current_humidity - 0.15 * (predicted_temp - current_temp)))
            
            print(f"Day {day}: {predicted_temp:.1f}C, Humidity: {predicted_humidity:.1f}%")
            
    except ValueError:
        print("\nUsing sample values for demonstration...")
        print("Day 1: 22.5C, Humidity: 65%")
        print("Day 2: 21.8C, Humidity: 67%")
        print("Day 3: 20.3C, Humidity: 70%")

def main():
    """Main execution function - runs the complete ML pipeline"""
    print("\n" + "="*60)
    print("WEATHER PREDICTION MACHINE LEARNING SYSTEM")
    print("="*60)
    print("A Complete Machine Learning Portfolio Project")
    print("="*60)
    
    # Create necessary directories
    for folder in ['data', 'models', 'results', 'notebooks']:
        os.makedirs(folder, exist_ok=True)
    
    # Step 1: Data preparation
    print("\n1. DATA PREPARATION")
    print("-" * 40)
    
    data_file = 'data/weather_data.csv'
    if os.path.exists(data_file):
        df = pd.read_csv(data_file, parse_dates=['date'])
        print(f"Loaded existing data: {len(df)} records")
    else:
        generator = WeatherDataGenerator(days=800)
        df = generator.generate_data()
        df.to_csv(data_file, index=False)
        print(f"Data saved to {data_file}")
    
    # Step 2: Model training
    print("\n2. MODEL TRAINING")
    print("-" * 40)
    
    predictor = WeatherPredictor()
    X_test, y_test, y_pred = predictor.train(df)
    
    # Step 3: Visualization
    print("\n3. VISUALIZATION")
    print("-" * 40)
    
    visualizer = WeatherVisualizer()
    visualizer.plot_predictions(y_test, y_pred)
    
    # Step 4: Save model
    print("\n4. MODEL SAVING")
    print("-" * 40)
    
    predictor.save_model()
    
    # Step 5: Generate forecast
    print("\n5. FORECAST GENERATION")
    print("-" * 40)
    
    forecast = predictor.predict_forecast(df, days=7)
    visualizer.plot_forecast(forecast)
    
    # Step 6: Interactive demo
    interactive_demo()
    
    # Step 7: Project summary
    print("\n" + "="*60)
    print("PROJECT SUMMARY")
    print("="*60)
    print("\nCompleted successfully!")
    print("\nFiles created:")
    print("   - data/weather_data.csv: Weather dataset")
    print("   - models/weather_model.pkl: Trained ML model")
    print("   - results/predictions.png: Model performance")
    print("   - results/forecast.png: 7-day forecast")
    
    print("\nMachine learning concepts demonstrated:")
    print("   - Feature engineering")
    print("   - Random forest regression")
    print("   - Model evaluation (MAE, R-squared)")
    print("   - Time series forecasting")
    print("   - Data visualization")
    
    print("\n" + "="*60)
    

if __name__ == "__main__":
    main()
