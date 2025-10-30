# Leaf Disease Detection Using Machine Learning

## Project Overview
This project focuses on developing a high-accuracy deep learning model for identifying **15 different classes of plant diseases** from leaf images. By automating plant disease detection, this solution helps farmers diagnose issues early, enabling timely treatment and reducing crop loss.  

The final model uses **Transfer Learning** with MobileNetV2 and achieved **91.13% validation accuracy**.

---

## Table of Contents
- [Project Objective](#project-objective)  
- [Technologies Used](#technologies-used)  
- [Dataset](#dataset)  
- [Methodology](#methodology)  
- [Results](#results)  
- [Model Deployment](#model-deployment)  
- [How to Use](#how-to-use)  
- [Conclusion](#conclusion)  

---

## Project Objective
The main goal of this project is to **automate plant disease detection** using computer vision. Early identification of plant diseases can help farmers take immediate action, minimizing crop damage and financial loss.

---

## Technologies Used
- **Platform:** Google Colab (with T4 GPU)  
- **Libraries:** TensorFlow, Keras, NumPy, Matplotlib  
- **Framework:** Convolutional Neural Networks (CNN) & Transfer Learning (MobileNetV2)  

---

## Dataset
- **Dataset Name:** PlantVillage  
- **Size:** 20,000+ images  
- **Classes:** 15 plant disease categories  
- **Description:** The dataset contains labeled leaf images for training and validation.  

---

## Methodology

### Step 1: Environment Setup & Data Preparation
- Mounted Google Drive to Colab  
- Unzipped dataset and verified directory structure  
- Organized images into class-wise folders for TensorFlow preprocessing  

### Step 2: Baseline CNN Model
- Built a **Sequential CNN** with three convolutional blocks  
- Trained for **15 epochs**  
- Achieved ~**79% accuracy**, highlighting the need for a more advanced approach  

### Step 3: Advanced Model Using Transfer Learning
- Implemented **MobileNetV2** pre-trained on ImageNet  
- Froze base weights and added a **custom classifier head**  
- Trained on the plant disease dataset  

### Step 4: Evaluation
- Achieved **91.13% validation accuracy**  
- Plotted **accuracy and loss curves**, confirming good learning trends without overfitting  

### Step 5: Model Saving & Prediction
- Saved model as `plant_disease_model.keras`  
- Tested on random images, achieving **>98% confidence** for correct predictions  

---

## Results
- **Baseline CNN Accuracy:** ~79%  
- **Transfer Learning Accuracy (MobileNetV2):** 91.13%  
- **Sample Prediction Confidence:** >98%  

The model demonstrated **robust performance** across all 15 classes and can be integrated into real-world applications.

---

## Model Deployment
The trained model is saved as `plant_disease_model.keras`. It can be deployed in:

- **Web applications** using Flask/Django  
- **Mobile applications** (Android/iOS) with TensorFlow Lite  
- **Embedded systems** for on-field disease detection  

---

## How to Use
1. Clone this repository  
2. Ensure you have **Python 3.8+** and required libraries installed (`TensorFlow`, `NumPy`, `Matplotlib`)  
3. Load the saved model:  
```python
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('plant_disease_model.keras')
  ```
## Preprocess your image and make predictions:
```python
from tensorflow.keras.preprocessing import image
import numpy as np

# Load and preprocess the image
img = image.load_img('sample_leaf.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
prediction = model.predict(img_array)
print("Predicted Class:", np.argmax(prediction))
  ```
## Conclusion
This project successfully developed a highly accurate plant disease detection model using Transfer Learning with MobileNetV2. With 91.13% validation accuracy, the model is reliable and ready for integration into real-world applications to help farmers make informed decisions.
