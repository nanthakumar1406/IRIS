# **README: Iris Species Classification with Random Forest and Neural Networks**  


[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/nanthakumar1406/IRIS/blob/main/iris.ipynb)
## **Project Overview**  
This project demonstrates **machine learning classification** on the **Iris dataset** using two different models:  
- **Random Forest Classifier** (scikit-learn)  
- **Neural Network** (TensorFlow/Keras)  

The goal is to predict the **species of an iris flower** based on its sepal and petal measurements.  

---

## **Features**  
- **Data Preprocessing:** Standard scaling and label encoding.  
- **Model Training:**  
  - Random Forest (scikit-learn)  
  - Neural Network (TensorFlow/Keras) with live training visualization.  
- **Evaluation:** Accuracy, classification report, and confusion matrix.  
- **Interactive Prediction:** Predict species from user input.  
- **Visualizations:**  
  - Training loss & accuracy curves (live updates).  
  - Feature importance (Random Forest).  
  - Species distribution in the dataset.  

---

## **Installation & Setup**  

### **Prerequisites**  
- Python 3.7+  
- Required libraries:  
  ```bash
  pip install pandas numpy scikit-learn tensorflow matplotlib seaborn tqdm
  ```

### **Run the Project**  
1. **Download the Iris dataset:**  
   - Ensure `iris.csv` is in the correct path (`C:/Users/Lenova/OneDrive/Report/iris/iris.csv`).  
   - Alternatively, modify the path in the code.  
2. **Run the script:**  
   ```bash
   python iris_classification.py
   ```

---

## **Usage**  

### **1. Training & Evaluation**  
- The script automatically:  
  - Preprocesses the data (scaling & encoding).  
  - Trains **Random Forest** and **Neural Network** models.  
  - Displays live training progress (loss & accuracy curves).  
  - Prints evaluation metrics (accuracy, classification report, confusion matrix).  

### **2. Interactive Prediction**  
After training, you can input flower measurements to get predictions:  
```
Enter the flower measurements to predict the species:
Sepal Length (cm): 5.1
Sepal Width (cm): 3.5
Petal Length (cm): 1.4
Petal Width (cm): 0.2

Random Forest Prediction: setosa
Neural Network Prediction: setosa
```

### **3. Visualizations**  
- **Training Curves:** Live-updating loss and accuracy graphs.  
- **Feature Importance:** Bar plot showing which features most influence predictions.  
- **Species Distribution:** Count plot of iris species in the dataset.  

---

## **Results**  
| Model               | Accuracy |  
|---------------------|----------|  
| Random Forest       | ~96.67%  |  
| Neural Network      | ~93.33%  |  

---

## **License**  
This project is licensed under the **MIT License**.  

---

### **Acknowledgments**  
- Dataset: [UCI Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)  
- Libraries: Pandas, scikit-learn, TensorFlow, Matplotlib, Seaborn  

🚀 **Happy Coding!**
