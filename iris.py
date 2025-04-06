import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

# Set page config
st.set_page_config(page_title="Iris Classification", layout="wide")

# Title
st.title("🌺 Iris Flower Classification")
st.markdown("""
This app compares Random Forest and Neural Network models for Iris species classification.
""")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    test_size = st.slider("Test Set Size (%)", 10, 40, 20)
    epochs = st.slider("Neural Network Epochs", 10, 200, 100)
    batch_size = st.slider("Batch Size", 1, 32, 5)

# Load data
@st.cache_data
def load_data():
    try:
        iris = pd.read_csv("iris.csv")
        iris.rename(columns={'species Name': 'species'}, inplace=True)
        return iris
    except FileNotFoundError:
        st.error("Iris dataset not found. Please upload iris.csv")
        return None

iris = load_data()

if iris is not None:
    # Preprocessing
    X = iris.iloc[:, :-1].values
    y = iris.iloc[:, -1].values
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42
    )

    # Model training
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Random Forest")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        y_pred_rf = rf_model.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        
        st.metric("Accuracy", f"{accuracy_rf*100:.2f}%")
        
        with st.expander("Classification Report"):
            st.text(classification_report(y_test, y_pred_rf, target_names=le.classes_))
        
        # Feature importance
        st.subheader("Feature Importance")
        fig, ax = plt.subplots()
        feature_importance = rf_model.feature_importances_
        ax.barh(range(len(feature_importance)), feature_importance, 
               tick_label=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
        st.pyplot(fig)

    with col2:
        st.subheader("Neural Network")
        model = keras.Sequential([
            keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(len(np.unique(y)), activation='softmax')
        ])
        
        model.compile(optimizer='adam', 
                     loss='sparse_categorical_crossentropy', 
                     metrics=['accuracy'])
        
        # Training progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        # Plot training history
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Loss Curve')
        ax1.legend()
        
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Accuracy Curve')
        ax2.legend()
        
        st.pyplot(fig)
        
        y_pred_nn = np.argmax(model.predict(X_test), axis=1)
        accuracy_nn = accuracy_score(y_test, y_pred_nn)
        st.metric("Accuracy", f"{accuracy_nn*100:.2f}%")

    # Confusion matrices
    st.subheader("Confusion Matrices")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Random Forest**")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred_rf), 
                    annot=True, fmt='d', cmap='Blues',
                    xticklabels=le.classes_, yticklabels=le.classes_)
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Neural Network**")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred_nn), 
                    annot=True, fmt='d', cmap='Greens',
                    xticklabels=le.classes_, yticklabels=le.classes_)
        st.pyplot(fig)

    # Visualizations
    st.subheader("Data Visualizations")
    
    tab1, tab2, tab3, tab4 = st.tabs(["PCA", "Distributions", "Violin", "Count"])
    
    with tab1:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        fig, ax = plt.subplots()
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], 
                        hue=le.inverse_transform(y), 
                        palette=['red', 'blue', 'green'])
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("PCA Visualization")
        st.pyplot(fig)
    
    with tab2:
        iris_melted = iris.melt(id_vars='species', var_name='Feature', value_name='Value')
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(x='Feature', y='Value', hue='species', 
                   data=iris_melted, palette="husl")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    with tab3:
        fig, ax = plt.subplots()
        sns.violinplot(x='species', y='sepal length (cm)', 
                       data=iris, palette=['red', 'blue', 'green'])
        st.pyplot(fig)
    
    with tab4:
        fig, ax = plt.subplots()
        sns.countplot(x='species', data=iris, palette='viridis')
        st.pyplot(fig)

    # Prediction interface
    st.subheader("🔮 Predict New Samples")
    
    with st.form("prediction_form"):
        cols = st.columns(4)
        sepal_length = cols[0].number_input("Sepal Length (cm)", min_value=0.0, value=5.1)
        sepal_width = cols[1].number_input("Sepal Width (cm)", min_value=0.0, value=3.5)
        petal_length = cols[2].number_input("Petal Length (cm)", min_value=0.0, value=1.4)
        petal_width = cols[3].number_input("Petal Width (cm)", min_value=0.0, value=0.2)
        
        submitted = st.form_submit_button("Predict Species")
        
        if submitted:
            input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            input_data = scaler.transform(input_data)
            
            # RF Prediction
            pred_rf = rf_model.predict(input_data)
            species_rf = le.inverse_transform(pred_rf)[0]
            
            # NN Prediction
            pred_nn = model.predict(input_data)
            species_nn = le.inverse_transform([np.argmax(pred_nn)])[0]
            
            st.success(f"Random Forest predicts: **{species_rf}**")
            st.success(f"Neural Network predicts: **{species_nn}**")

# Footer
st.markdown("---")
st.markdown("Iris classification using machine learning | [GitHub Repo](#)")