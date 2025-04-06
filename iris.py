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
import plotly.express as px

# Set page config
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌺",
    layout="wide"
)
# Title
st.title("🌺 Iris Flower Classification")
st.markdown("""
This app compares Random Forest and Neural Network models for Iris species classification.
""")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    test_size = st.slider("Test Set Size (%)", 10, 40, 20)
    epochs = st.slider("Neural Network Epochs", 10, 200, 50)
    batch_size = st.slider("Batch Size", 1, 32, 5)

# Load data
@st.cache_data
def load_data():
    try:
        iris = pd.read_csv("iris.csv")
        # Verify and standardize column names
        iris.columns = iris.columns.str.lower().str.replace(' ', '_')
        if 'species_name' in iris.columns:
            iris.rename(columns={'species_name': 'species'}, inplace=True)
        elif 'species' not in iris.columns:
            st.error("No 'species' column found in dataset")
            return None
        return iris
    except FileNotFoundError:
        st.error("Iris dataset not found. Please upload iris.csv")
        return None

iris = load_data()

if iris is not None:
    st.subheader("📊 Dataset Overview")
    
    with st.expander("View Raw Data"):
        st.dataframe(iris)
    # Debug: Show column names
    st.write("Dataset columns:", iris.columns.tolist())
    
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
               tick_label=[col.replace('_', ' ').title() for col in iris.columns[:-1]])
        st.pyplot(fig)

    with col2:
        st.subheader("Neural Network")
        
        # Model definition
        model = keras.Sequential([
            keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
            keras.layers.Dense(8, activation='relu'),
            keras.layers.Dense(len(np.unique(y)), activation='softmax')
        ])
        
        model.compile(optimizer='adam', 
                     loss='sparse_categorical_crossentropy', 
                     metrics=['accuracy'])
        
        # Model summary
        with st.expander("Model Architecture"):
            summary_list = []
            model.summary(print_fn=lambda x: summary_list.append(x))
            st.text("\n".join(summary_list))
        
        # Training
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        # Make predictions
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred_nn = np.argmax(y_pred_probs, axis=1)  # This defines y_pred_nn
        
        # Now we can safely use y_pred_nn
        accuracy_nn = accuracy_score(y_test, y_pred_nn)
        st.metric("Accuracy", f"{accuracy_nn*100:.2f}%")
        
        # Classification report
        with st.expander("Detailed Classification Report"):
            report = classification_report(y_test, y_pred_nn, target_names=le.classes_, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))
        
        # ROC Curve
        st.write("### ROC Curves")
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc
        
        y_test_bin = label_binarize(y_test, classes=[0,1,2])
        n_classes = y_test_bin.shape[1]
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        fig, ax = plt.subplots(figsize=(8,6))
        colors = ['blue', 'red', 'green']
        for i, color in zip(range(n_classes), colors):
            ax.plot(fpr[i], tpr[i], color=color,
                    label=f'ROC curve (class {le.classes_[i]}) (area = {roc_auc[i]:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic')
        ax.legend(loc="lower right")
        st.pyplot(fig)
        
        # Training history plots
        st.write("### Training History")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
        
        ax1.plot(history.history['loss'], label='Train')
        ax1.plot(history.history['val_loss'], label='Validation')
        ax1.set_title('Loss History')
        ax1.set_ylabel('Loss')
        ax1.set_xlabel('Epoch')
        ax1.legend()
        
        ax2.plot(history.history['accuracy'], label='Train')
        ax2.plot(history.history['val_accuracy'], label='Validation')
        ax2.set_title('Accuracy History')
        ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        
        st.pyplot(fig)
        
    
        

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
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["PCA", "Distributions", "Violin", "Count", "Column Chart"])

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
        try:
            iris_melted = iris.melt(
                id_vars='species', 
                var_name='Feature', 
                value_name='Value'
            )
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.boxplot(
                x='Feature', 
                y='Value', 
                hue='species', 
                data=iris_melted, 
                palette="husl"
            )
            plt.xticks(rotation=45)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating distribution plot: {str(e)}")
            st.write("Available columns:", iris.columns.tolist())
    
    with tab3:
        try:
            fig, ax = plt.subplots()
            sns.violinplot(
                x='species', 
                y=iris.columns[0],  # Use first feature column
                data=iris, 
                palette=['red', 'blue', 'green']
            )
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating violin plot: {str(e)}")
    
    with tab4:
        try:
            fig, ax = plt.subplots()
            sns.countplot(
                x='species', 
                data=iris, 
                palette='viridis'
            )
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating count plot: {str(e)}")
            
    with tab5:
        try:
            # Get predictions for all test samples
            y_pred_probs = model.predict(X_test, verbose=0)
            avg_probs = np.mean(y_pred_probs, axis=0)
            
            fig, ax = plt.subplots()
            ax.bar(le.classes_, avg_probs, color=['blue', 'red', 'green'])
            ax.set_title("Average Prediction Probabilities")
            ax.set_ylabel("Probability")
            ax.set_ylim([0,1])
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating column chart: {str(e)}")

    # Prediction interface
    st.subheader("🔮 Predict New Samples")
    
    with st.form("prediction_form"):
        feature_names = [col for col in iris.columns if col != 'species']
        cols = st.columns(len(feature_names))
        
        input_values = []
        for i, col in enumerate(feature_names):
            input_values.append(
                cols[i].number_input(
                    f"{col.replace('_', ' ').title()} (cm)",
                    min_value=0.0,
                    value=float(iris[col].median())
                )
            )
        
        submitted = st.form_submit_button("Predict Species")
        
        if submitted:
            input_data = np.array([input_values])
            input_data = scaler.transform(input_data)
            
            # RF Prediction
            pred_rf = rf_model.predict(input_data)
            species_rf = le.inverse_transform(pred_rf)[0]
            
            # NN Prediction
            pred_nn = model.predict(input_data, verbose=0)
            species_nn = le.inverse_transform([np.argmax(pred_nn)])[0]
            
            st.success(f"Random Forest predicts: **{species_rf}**")
            st.success(f"Neural Network predicts: **{species_nn}**")

# Footer
st.markdown("---")
st.markdown("Iris classification using machine learning")