"""
SmartSkin Interactive Dashboard
Real-time emotion monitoring visualization
"""

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime

# Page config
st.set_page_config(page_title="SmartSkin Simulator", page_icon="👕", layout="wide")

# Emotion profiles
EMOTION_PROFILES = {
    'Calm': {'hr_range': (60, 75), 'temp_range': (32.0, 34.0), 'color': 'rgb(0, 0, 255)', 'hex': '#0000FF'},
    'Happy': {'hr_range': (70, 90), 'temp_range': (33.0, 35.0), 'color': 'rgb(255, 255, 0)', 'hex': '#FFFF00'},
    'Stressed': {'hr_range': (90, 110), 'temp_range': (36.0, 38.0), 'color': 'rgb(255, 0, 0)', 'hex': '#FF0000'},
    'Excited': {'hr_range': (100, 140), 'temp_range': (36.0, 38.0), 'color': 'rgb(255, 128, 0)', 'hex': '#FF8000'},
    'Focused': {'hr_range': (70, 85), 'temp_range': (33.0, 34.0), 'color': 'rgb(0, 255, 0)', 'hex': '#00FF00'},
    'Anxious': {'hr_range': (95, 120), 'temp_range': (35.0, 37.0), 'color': 'rgb(128, 0, 128)', 'hex': '#800080'},
    'Relaxed': {'hr_range': (55, 70), 'temp_range': (31.0, 33.0), 'color': 'rgb(173, 216, 230)', 'hex': '#ADD8E6'},
    'Angry': {'hr_range': (110, 150), 'temp_range': (37.0, 39.0), 'color': 'rgb(139, 0, 0)', 'hex': '#8B0000'},
    'Sad': {'hr_range': (65, 80), 'temp_range': (32.0, 34.0), 'color': 'rgb(70, 70, 70)', 'hex': '#464646'},
    'Energetic': {'hr_range': (85, 120), 'temp_range': (34.0, 36.0), 'color': 'rgb(255, 0, 255)', 'hex': '#FF00FF'}
}

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'running' not in st.session_state:
    st.session_state.running = False

def generate_data(emotion, num_samples=200):
    """Generate training data"""
    profile = EMOTION_PROFILES[emotion]
    hr_min, hr_max = profile['hr_range']
    temp_min, temp_max = profile['temp_range']
    
    hrs = np.random.uniform(hr_min, hr_max, num_samples) + np.random.normal(0, 2, num_samples)
    temps = np.random.uniform(temp_min, temp_max, num_samples) + np.random.normal(0, 0.3, num_samples)
    
    hrs = np.clip(hrs, 40, 200)
    temps = np.clip(temps, 30, 42)
    
    data = []
    for hr, temp in zip(hrs, temps):
        data.append({
            'heart_rate': hr,
            'temperature': temp,
            'hr_variability': abs(hr - 70) / 70,
            'temp_trend': temp - 33.0,
            'emotion': emotion
        })
    return data

@st.cache_resource
def train_model():
    """Train the emotion detection model"""
    data = []
    for emotion in EMOTION_PROFILES.keys():
        data.extend(generate_data(emotion, 200))
    
    df = pd.DataFrame(data)
    X = df[['heart_rate', 'temperature', 'hr_variability', 'temp_trend']]
    y = df['emotion']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, model.score(X_scaled, y)

# Main UI
st.title("👕 SmartSkin Emotion Detection Simulator")
st.markdown("### Real-time emotion monitoring with adaptive LED feedback")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    if not st.session_state.model_trained:
        if st.button("🧠 Train Model", use_container_width=True):
            with st.spinner("Training emotion detection model..."):
                model, scaler, accuracy = train_model()
                st.session_state.model = model
                st.session_state.scaler = scaler
                st.session_state.accuracy = accuracy
                st.session_state.model_trained = True
                st.success(f"✅ Model trained! Accuracy: {accuracy:.2%}")
    
    if st.session_state.model_trained:
        st.success(f"✅ Model ready (Acc: {st.session_state.accuracy:.2%})")
        
        st.markdown("---")
        simulation_mode = st.radio("Simulation Mode:", 
                                   ["Manual Control", "Auto Simulation"])
        
        if simulation_mode == "Manual Control":
            selected_emotion = st.selectbox("Select Emotion:", 
                                           list(EMOTION_PROFILES.keys()))
            
            if st.button("📊 Generate Reading", use_container_width=True):
                profile = EMOTION_PROFILES[selected_emotion]
                hr = np.random.uniform(*profile['hr_range']) + np.random.normal(0, 2)
                temp = np.random.uniform(*profile['temp_range']) + np.random.normal(0, 0.3)
                
                features = np.array([[hr, temp, abs(hr-70)/70, temp-33.0]])
                features_scaled = st.session_state.scaler.transform(features)
                
                predicted = st.session_state.model.predict(features_scaled)[0]
                confidence = np.max(st.session_state.model.predict_proba(features_scaled))
                
                reading = {
                    'timestamp': datetime.now(),
                    'heart_rate': hr,
                    'temperature': temp,
                    'actual': selected_emotion,
                    'predicted': predicted,
                    'confidence': confidence
                }
                
                st.session_state.history.append(reading)
        
        else:  # Auto Simulation
            interval = st.slider("Update Interval (sec):", 1, 10, 3)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ Start", use_container_width=True):
                    st.session_state.running = True
            with col2:
                if st.button("⏸️ Stop", use_container_width=True):
                    st.session_state.running = False
        
        st.markdown("---")
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# Main content area
if st.session_state.model_trained and len(st.session_state.history) > 0:
    latest = st.session_state.history[-1]
    
    # Current reading display
    st.markdown("### 📡 Current Reading")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("❤️ Heart Rate", f"{latest['heart_rate']:.1f} BPM")
    with col2:
        st.metric("🌡️ Temperature", f"{latest['temperature']:.1f}°C")
    with col3:
        st.metric("🎭 Predicted Emotion", latest['predicted'])
    with col4:
        st.metric("✅ Confidence", f"{latest['confidence']:.1%}")
    
    # LED Visualization
    st.markdown("### 💡 LED Status")
    led_color = EMOTION_PROFILES[latest['predicted']]['hex']
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 20px;">
        <div style="width: 100px; height: 100px; background-color: {led_color}; 
                    border-radius: 50%; box-shadow: 0 0 30px {led_color};
                    animation: pulse 2s infinite;">
        </div>
        <div>
            <h2 style="margin: 0;">{latest['predicted']}</h2>
            <p style="color: gray; margin: 0;">LED Color: {led_color}</p>
        </div>
    </div>
    <style>
        @keyframes pulse {{
            0%, 100% {{ transform: scale(1); }}
            50% {{ transform: scale(1.1); }}
        }}
    </style>
    """, unsafe_allow_html=True)
    
    # Charts
    if len(st.session_state.history) > 1:
        st.markdown("### 📊 Real-time Metrics")
        
        df_hist = pd.DataFrame(st.session_state.history)
        
        # Heart rate chart
        fig_hr = go.Figure()
        fig_hr.add_trace(go.Scatter(
            y=df_hist['heart_rate'],
            mode='lines+markers',
            name='Heart Rate',
            line=dict(color='red', width=2)
        ))
        fig_hr.update_layout(
            title="Heart Rate Over Time",
            yaxis_title="BPM",
            height=300,
            template="plotly_white"
        )
        st.plotly_chart(fig_hr, use_container_width=True)
        
        # Temperature chart
        fig_temp = go.Figure()
        fig_temp.add_trace(go.Scatter(
            y=df_hist['temperature'],
            mode='lines+markers',
            name='Temperature',
            line=dict(color='orange', width=2)
        ))
        fig_temp.update_layout(
            title="Skin Temperature Over Time",
            yaxis_title="°C",
            height=300,
            template="plotly_white"
        )
        st.plotly_chart(fig_temp, use_container_width=True)
        
        # Emotion timeline
        st.markdown("### 🎭 Emotion Timeline")
        emotion_colors = [EMOTION_PROFILES[e]['hex'] for e in df_hist['predicted']]
        
        fig_emotions = go.Figure()
        fig_emotions.add_trace(go.Bar(
            y=df_hist.index,
            x=[1] * len(df_hist),
            orientation='h',
            marker=dict(color=emotion_colors),
            text=df_hist['predicted'],
            textposition='inside',
            hovertemplate='<b>%{text}</b><br>Confidence: %{customdata:.1%}<extra></extra>',
            customdata=df_hist['confidence']
        ))
        fig_emotions.update_layout(
            height=400,
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis_title="Reading #",
            template="plotly_white"
        )
        st.plotly_chart(fig_emotions, use_container_width=True)
        
        # Statistics
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📈 Statistics")
            correct = (df_hist['actual'] == df_hist['predicted']).sum()
            st.metric("Accuracy", f"{correct/len(df_hist):.1%}")
            st.metric("Avg Confidence", f"{df_hist['confidence'].mean():.1%}")
        
        with col2:
            st.markdown("### 🎯 Emotion Distribution")
            emotion_dist = df_hist['predicted'].value_counts()
            fig_pie = px.pie(values=emotion_dist.values, names=emotion_dist.index,
                            color=emotion_dist.index,
                            color_discrete_map={e: EMOTION_PROFILES[e]['hex'] 
                                               for e in EMOTION_PROFILES.keys()})
            st.plotly_chart(fig_pie, use_container_width=True)

elif not st.session_state.model_trained:
    st.info("👈 Click 'Train Model' in the sidebar to get started!")

else:
    st.info("Generate your first reading using the sidebar controls!")

# Auto simulation loop
if st.session_state.get('running', False) and st.session_state.model_trained:
    time.sleep(interval)
    
    emotion = np.random.choice(list(EMOTION_PROFILES.keys()))
    profile = EMOTION_PROFILES[emotion]
    
    hr = np.random.uniform(*profile['hr_range']) + np.random.normal(0, 2)
    temp = np.random.uniform(*profile['temp_range']) + np.random.normal(0, 0.3)
    
    features = np.array([[hr, temp, abs(hr-70)/70, temp-33.0]])
    features_scaled = st.session_state.scaler.transform(features)
    
    predicted = st.session_state.model.predict(features_scaled)[0]
    confidence = np.max(st.session_state.model.predict_proba(features_scaled))
    
    reading = {
        'timestamp': datetime.now(),
        'heart_rate': hr,
        'temperature': temp,
        'actual': emotion,
        'predicted': predicted,
        'confidence': confidence
    }
    
    st.session_state.history.append(reading)
    st.rerun()
