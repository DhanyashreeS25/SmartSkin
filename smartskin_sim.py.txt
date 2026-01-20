# Import all the tools we need
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
import time
from datetime import datetime
import plotly.graph_objects as go

# Emotion definitions (based on your research paper)
EMOTION_PROFILES = {
    'Calm': {'hr_range': (60, 75), 'temp_range': (32.0, 34.0), 'color': (0, 0, 255)},      # Blue
    'Happy': {'hr_range': (70, 90), 'temp_range': (33.0, 35.0), 'color': (255, 255, 0)},   # Yellow
    'Stressed': {'hr_range': (90, 110), 'temp_range': (36.0, 38.0), 'color': (255, 0, 0)}, # Red
    'Excited': {'hr_range': (100, 140), 'temp_range': (36.0, 38.0), 'color': (255, 128, 0)}, # Orange
    'Focused': {'hr_range': (70, 85), 'temp_range': (33.0, 34.0), 'color': (0, 255, 0)},   # Green
    'Anxious': {'hr_range': (95, 120), 'temp_range': (35.0, 37.0), 'color': (128, 0, 128)}, # Purple
    'Relaxed': {'hr_range': (55, 70), 'temp_range': (31.0, 33.0), 'color': (173, 216, 230)}, # Light Blue
    'Angry': {'hr_range': (110, 150), 'temp_range': (37.0, 39.0), 'color': (139, 0, 0)},   # Dark Red
    'Sad': {'hr_range': (65, 80), 'temp_range': (32.0, 34.0), 'color': (70, 70, 70)},      # Gray
    'Energetic': {'hr_range': (85, 120), 'temp_range': (34.0, 36.0), 'color': (255, 0, 255)} # Magenta
}

def generate_sensor_data(emotion, num_samples=20):
    """
    This function creates FAKE sensor readings
    Like a simulator pretending to be real sensors!
    """
    profile = EMOTION_PROFILES[emotion]
    
    # Generate random heart rates within the emotion's range
    hr_min, hr_max = profile['hr_range']
    heart_rates = np.random.uniform(hr_min, hr_max, num_samples)
    
    # Generate random temperatures within the emotion's range
    temp_min, temp_max = profile['temp_range']
    temperatures = np.random.uniform(temp_min, temp_max, num_samples)
    
    # Add realistic noise (sensors aren't perfect!)
    heart_rates += np.random.normal(0, 2, num_samples)  # ±2 BPM noise
    temperatures += np.random.normal(0, 0.3, num_samples)  # ±0.3°C noise
    
    return heart_rates, temperatures

def create_dataset():
    """
    Build a complete dataset with 200 samples per emotion
    (Just like your paper says!)
    """
    data = []
    
    for emotion in EMOTION_PROFILES.keys():
        # Generate 200 samples for each emotion
        hrs, temps = generate_sensor_data(emotion, num_samples=200)
        
        for hr, temp in zip(hrs, temps):
            # Calculate features (like your paper)
            hr_variability = abs(hr - 70) / 70  # Compare to baseline
            temp_trend = temp - 33.0  # Deviation from baseline
            
            data.append({
                'heart_rate': hr,
                'temperature': temp,
                'hr_variability': hr_variability,
                'temp_trend': temp_trend,
                'emotion': emotion
            })
    
    # Convert to pandas DataFrame (like an Excel sheet)
    df = pd.DataFrame(data)
    return df

def train_emotion_model(df):
    """
    Teach the computer to recognize emotions
    (This is the Machine Learning part!)
    """
    # Prepare features (X) and labels (y)
    X = df[['heart_rate', 'temperature', 'hr_variability', 'temp_trend']]
    y = df['emotion']
    
    # Split into training and testing sets (80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalize the data (make all values similar scale)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train the Random Forest model (100 trees!)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Check accuracy
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)
    
    print(f"Training Accuracy: {train_accuracy:.3f}")
    print(f"Testing Accuracy: {test_accuracy:.3f}")
    
    return model, scaler

