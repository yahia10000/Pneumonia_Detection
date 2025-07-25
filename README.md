# 🩺 Pneumonia Detection from Chest X-Ray

This project uses a Convolutional Neural Network (CNN) to detect **Pneumonia** from grayscale chest X-ray images. It includes data preprocessing, augmentation, model training, evaluation, and deployment using **Streamlit** for an interactive web app.

---

## 📘 Notebook Overview

The notebook walks through the following steps:

### 1. **Import Libraries**
Essential libraries are imported, including TensorFlow, Keras, Matplotlib, NumPy, and others.

### 2. **Load and Explore Data**
- The dataset is loaded.
- Image dimensions, sample shapes, and class distribution are explored.

### 3. **Data Preprocessing**
- Normalization and reshaping of image arrays.
- One-hot encoding of labels.

### 4. **Train-Test Split**
- Dataset split into training, validation, and test sets.

### 5. **Data Augmentation**
To address class imbalance and improve generalization, image augmentation techniques like:
- Rotation
- Zoom
- Horizontal flip  
are applied using `ImageDataGenerator`.

### 6. **Model Architecture**
- CNN built using `Sequential` API.
- Layers include Conv2D, MaxPooling, Flatten, Dense, Dropout.

### 7. **Model Compilation & Training**
- Loss: `categorical_crossentropy`
- Optimizer: `adam`
- Metrics: `accuracy`
- Training with callbacks such as EarlyStopping and ModelCheckpoint.

### 8. **Evaluation**
- Accuracy and loss plots.
- Classification report and confusion matrix on the test set.

### 9. **Save and Load Model**
- The trained model is saved in `.h5` format.
- Later reloaded for inference.

### 10. **Streamlit Deployment**
- A Streamlit app is created to upload X-ray images and display predictions interactively.

---

## 🚀 Live Demo

Check out the live deployed version of the model on **Streamlit Cloud**:

👉 [**Live Demo**](https://pneumoniadetection-vdmau5rqx8zhtv4dgfplsi.streamlit.app/)

> _Make sure to update the above link with your actual Streamlit app URL._

---

## 🖼️ Sample Images

### 📷 Normal Chest X-ray
![Normal X-ray](images/normal_sample.png)

### 🦠 Pneumonia Chest X-ray
![Pneumonia X-ray](images/pneumonia_sample.png)

> _Place your image samples inside the `images/` folder._

---

## 🧑‍💻 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/your-username/pneumonia-detection.git
cd pneumonia-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
````

---

## 📦 Requirements

* Python 3.7+
* TensorFlow / Keras
* NumPy
* Matplotlib
* scikit-learn
* Streamlit
* Pillow

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 📩 Contact

For questions or collaborations, feel free to reach out!


