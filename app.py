import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Earthquake Magnitude Prediction",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and header
st.title("🌍 Climate Risk & Disaster Management")
st.subheader("Earthquake Magnitude Prediction System")
st.markdown("---")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Home", "Data Analysis", "Model Prediction", "Visualization"])

# Load data function
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('earthquake.csv')
        return data
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure earthquake.csv is in the same directory.")
        return None

# Load model function
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.warning("Pre-trained model not found. Training a new model...")
        return None

# Main application logic
def main():
    data = load_data()
    
    if data is not None:
        if page == "Home":
            show_home_page(data)
        elif page == "Data Analysis":
            show_data_analysis(data)
        elif page == "Model Prediction":
            show_prediction_page(data)
        elif page == "Visualization":
            show_visualization_page(data)

def show_home_page(data):
    st.markdown("## Project Overview")
    st.write("""
    This application predicts earthquake magnitudes using machine learning techniques.
    The project analyzes seismic data to identify patterns and make predictions for disaster management.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(data))
    with col2:
        st.metric("Features", len(data.columns) - 1)
    with col3:
        st.metric("Avg Magnitude", f"{data['magnitude'].mean():.2f}")
    
    st.markdown("## Dataset Preview")
    st.dataframe(data.head())

def show_data_analysis(data):
    st.markdown("## Exploratory Data Analysis")
    
    # Basic statistics
    st.subheader("Dataset Statistics")
    st.write(data.describe())
    
    # Missing values
    st.subheader("Missing Values")
    missing_data = data.isnull().sum()
    if missing_data.sum() > 0:
        st.write(missing_data[missing_data > 0])
    else:
        st.success("No missing values found!")
    
    # Correlation matrix
    st.subheader("Feature Correlations")
    numeric_data = data.select_dtypes(include=[np.number])
    if len(numeric_data.columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

def show_prediction_page(data):
    st.markdown("## Earthquake Magnitude Prediction")
    
    # Load or train model
    model = load_model()
    
    if model is None:
        # Train a simple model for demonstration
        st.info("Training a new Random Forest model...")
        numeric_features = data.select_dtypes(include=[np.number]).columns.tolist()
        if 'magnitude' in numeric_features:
            numeric_features.remove('magnitude')
        
        if len(numeric_features) > 0:
            X = data[numeric_features].fillna(0)
            y = data['magnitude']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # Model performance
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            st.success(f"Model trained! R² Score: {r2:.4f}, MSE: {mse:.4f}")
            
            # Save model
            with open('model.pkl', 'wb') as f:
                pickle.dump(model, f)
        else:
            st.error("No numeric features found for training.")
            return
    
    # Input form for prediction
    st.subheader("Make a Prediction")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        # Example input fields (adjust based on your actual features)
        with col1:
            latitude = st.number_input("Latitude", value=0.0, format="%.6f")
            longitude = st.number_input("Longitude", value=0.0, format="%.6f")
        
        with col2:
            depth = st.number_input("Depth (km)", value=10.0, min_value=0.0)
            # Add more input fields based on your dataset features
        
        submitted = st.form_submit_button("Predict Magnitude")
        
        if submitted and model is not None:
            # Prepare input data (adjust based on your model features)
            input_data = np.array([[latitude, longitude, depth]])  # Add more features as needed
            prediction = model.predict(input_data)
            
            st.success(f"Predicted Earthquake Magnitude: {prediction[0]:.2f}")
            
            # Risk assessment
            if prediction[0] < 3.0:
                st.info("🟢 Low Risk: Minor earthquake")
            elif prediction[0] < 5.0:
                st.warning("🟡 Moderate Risk: Noticeable earthquake")
            elif prediction[0] < 7.0:
                st.error("🟠 High Risk: Strong earthquake")
            else:
                st.error("🔴 Very High Risk: Major earthquake")

def show_visualization_page(data):
    st.markdown("## Data Visualizations")
    
    # Magnitude distribution
    st.subheader("Magnitude Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(data['magnitude'], bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Earthquake Magnitudes')
    st.pyplot(fig)
    
    # Additional visualizations based on your data
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) > 1:
        st.subheader("Feature Relationships")
        feature1 = st.selectbox("Select X-axis feature", numeric_columns)
        feature2 = st.selectbox("Select Y-axis feature", numeric_columns, index=1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(data[feature1], data[feature2], alpha=0.6)
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
        ax.set_title(f'{feature1} vs {feature2}')
        st.pyplot(fig)

if __name__ == "__main__":
    main()