# LungCancerDetection


#  Lung Cancer Detection Using Deep Learning

This project focuses on building a **deep learning-based system** to detect **lung cancer** from **CT scan images**. The goal is to leverage Convolutional Neural Networks (CNN) and transfer learning models such as **ResNet50**, **VGG16**, and **MobileNetV2** to assist in early cancer detection and classification.

---

##  Features

*  Handles large-scale CT scan image datasets
*  Supports multiple CNN architectures (ResNet50, VGG16, MobileNetV2)
*  Detects and classifies cancerous vs non-cancerous lung tissues
*  Visualizes model performance (accuracy, confusion matrix, loss curves)
*  Includes trained models for quick inference
*  Modular code — easy to extend or retrain with custom datasets

---

##  Project Structure

```
LungCancerDetection/
│
├── data/                     # Dataset (not included in repo)
│   ├── train/
│   ├── test/
│
├── models/                   # Saved trained models (.pkl / .h5)
│   ├── model_resnet50.pkl
│   ├── model_vgg16.pkl
│
├── notebooks/                # Jupyter notebooks for training & analysis
│   ├── lung_cancer_training.ipynb
│
├── src/                      # Python source code
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

##  Models Used

| Model            | Description                                         | Framework        |
| ---------------- | --------------------------------------------------- | ---------------- |
| **CNN (Custom)** | Basic convolutional architecture built from scratch | TensorFlow/Keras |
| **ResNet50**     | Deep residual network (50 layers) for high accuracy | TensorFlow/Keras |
| **VGG16**        | Simpler deep CNN architecture                       | TensorFlow/Keras |
| **MobileNetV2**  | Lightweight CNN for mobile deployment               | TensorFlow/Keras |

---

##  Tech Stack

* **Language:** Python
* **Libraries:** TensorFlow, Keras, NumPy, OpenCV, Matplotlib, Scikit-learn
* **IDE:** Jupyter Notebook / VS Code
* **Frameworks:** Deep Learning (CNN, Transfer Learning)

---

##  Installation and Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/gschandrasekhar/LungCancerDetection.git
   cd LungCancerDetection
   ```

2. **Create a virtual environment (optional)**

   ```bash
   python -m venv venv
   venv\Scripts\activate    # On Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebook or script**

   ```bash
   jupyter notebook notebooks/lung_cancer_training.ipynb
   ```

   or

   ```bash
   python src/train_model.py
   ```

---

##  Results

| Model        | Accuracy | Notes                                   |
| ------------ | -------- | --------------------------------------- |
| CNN (Custom) | 85%      | Baseline model                          |
| VGG16        | 91%      | Moderate accuracy, higher training time |
| ResNet50     | 94%      | Best performing model                   |
| MobileNetV2  | 92%      | Lightweight, good trade-off             |

*(Results may vary depending on dataset and training parameters.)*

---

##  Future Work

* Deploy model as a **web app using Streamlit or Flask**
* Integrate **Grad-CAM visualization** for explainability
* Add **real-time prediction** from uploaded CT images
* Optimize for mobile and edge devices

---

##  References

* Kaggle: [Lung Cancer CT Scan Dataset](https://www.kaggle.com/)
* Research Paper: “Deep Learning for Lung Cancer Detection Using CT Scans”
* TensorFlow / Keras Documentation

---

## 🧑‍💻 Author

**Chandra Sekhar G**
📧 [Email Me](mailto:gschandrasekhar@gmail.com)
🌐 [LinkedIn](https://www.linkedin.com/in/gschandrasekhar/)
💻 [GitHub](https://github.com/gschandrasekhar)

---

## 🪪 License

This project is licensed under the **MIT License** — feel free to use and modify for research or learning purposes.


