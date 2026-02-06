# 🚗 Car vs Bike Image Classification using Deep Learning (CNN)

### **Automated Vehicle Detection with TensorFlow & Keras**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)]()

---

## 📖 Overview

**Visual recognition is at the core of modern automation.**

This project builds a robust **Convolutional Neural Network (CNN)** capable of classifying images as either a **Car** or a **Bike**. Built with **TensorFlow** and **Keras**, the model learns to identify distinct visual features directly from raw image data.

To make the technology accessible, the model is deployed as an interactive web application using **Streamlit**, allowing users to upload images and get instant predictions. 🧠📸

> *"Teaching machines to see the difference between four wheels and two."*

---

## 🚀 What the App Does

This application serves as an end-to-end image classifier:

* **🔍 Binary Classification:** accurately categorizes vehicle images into "Car" or "Bike".
* **🧠 Feature Learning:** Automatically extracts visual patterns like edges, shapes, and textures without manual feature engineering.
* **🔄 Robust Handling:** Capable of processing images with varied sizes, lighting conditions, and orientations.
* **⚡ Real-Time Predictions:** Delivers fast and reliable results through a clean, user-friendly interface.

---

## 🛠️ Tools & Technologies Used

* **🐍 Python:** The core programming language for model logic.
* **🧠 TensorFlow / Keras:** Used for constructing and training the Deep Learning CNN architecture.
* **🖼️ ImageDataGenerator:** Applied for image preprocessing and data augmentation to prevent overfitting.
* **🧮 NumPy:** Used for efficient numerical operations and array handling.
* **🌐 Streamlit:** Used to build the frontend web application for easy deployment.

---

## 📊 Model Performance

The model was evaluated on unseen test data to ensure reliability:

* **✅ Accuracy:** Achieved high classification accuracy on the validation set.
* **🎯 Precision:** Delivers reliable positive predictions with minimal false positives.
* **🔎 Recall:** Demonstrates a strong ability to detect both vehicle classes (low false negative rate).
* **📌 Balance:** The model maintains consistent performance across both classes, ensuring no bias toward one vehicle type.

---

## 💡 Key Insights

* **Patterns Matter:** CNNs are highly effective at capturing vehicle-specific hierarchies (e.g., wheels vs. handlebars).
* **Augmentation is Key:** Using `ImageDataGenerator` significantly improved the model's ability to generalize to real-world data.
* **Depth Wins:** Deeper convolution layers allowed the model to distinguish more complex object features.
* **Real-World Utility:** This architecture is scalable for use cases like traffic monitoring, automated toll collection, and parking management.

---

## ⚙️ The Workflow

The pipeline moves from raw data ingestion to a deployed prediction interface.

```mermaid
graph TD;
    A["📷 User Uploads Image\n(Car or Bike)"] -->|Preprocessing| B{"⚙️ ImageDataGenerator\nRescaling & Formatting"};
    B -->|Input Tensor| C["🧠 CNN Model\n(TensorFlow/Keras)"];
    C -->|Feature Extraction| D["🔍 Convolution & Pooling Layers"];
    D -->|Classification| E["⚡ Output Prediction\n(Probability Score)"];
    E -->|Display| F["🌐 Streamlit UI\nFinal Result"];

    style A fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style F fill:#bfb,stroke:#333,stroke-width:2px
