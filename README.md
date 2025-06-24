# Diabetes Prediction Web App

This is a simple and interactive **machine learning web application** built using **Streamlit**. The model predicts whether a person is likely to have diabetes based on certain health parameters. It is one of my beginner-level ML projects that demonstrates the use of classification models in healthcare.

## 🎯 Objective

To build and deploy a machine learning model that can predict the likelihood of diabetes based on input medical data using classical ML techniques.

## 💡 Features

- 🩺 Predicts diabetes based on user inputs (like glucose level, BMI, age, etc.)
- 📊 Real-time and interactive interface using Streamlit
- ✅ Lightweight and beginner-friendly
- 🚀 Deployable on local server or platforms like Streamlit Cloud

## 🧠 Model Overview

- **Model**: Logistic Regression / Random Forest / Decision Tree (depending on training)
- **Trained On**: PIMA Indians Diabetes Dataset (commonly used dataset)
- **Saved As**: `diabetes_model.sav` or similar
- **Vectorization**: Not applicable (tabular data used directly)

## 📁 Project Structure

```
DiabetesApp/
├── streamlit_diabetes.py        # Streamlit web app
├── diabetes_model.sav           # Pre-trained ML model
└── README.md                    # Project documentation
```

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **scikit-learn**
- **Pandas**
- **Joblib / Pickle**

## 🔧 Installation & Running

### Step 1: Clone the Repository
```bash
git clone https://github.com/ayush123-bit/DiabetesApp.git
cd DiabetesApp
```

### Step 2: Install Dependencies
If `requirements.txt` is provided:
```bash
pip install -r requirements.txt
```
Or install manually:
```bash
pip install streamlit scikit-learn pandas joblib
```

### Step 3: Run the Streamlit App
```bash
streamlit run streamlit_diabetes.py
```

Then open your browser at: [http://localhost:8501](http://localhost:8501)

## 🧾 Input Features

Typical input fields expected from the user:
- Number of Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin Level
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age

> These values are then passed to the ML model which returns a prediction.

## ✅ Example Output

- Input:  
  `Glucose: 140, BMI: 32.0, Age: 45, etc.`  
- Output:  
  `The model predicts: You are likely to have diabetes.`

## 📚 Future Improvements

- Add visualization of input/output
- Display model metrics (accuracy, confusion matrix, etc.)
- Allow CSV upload for batch predictions
- Enable model selection and retraining

## 🙋 Author

**Ayush Rai**  
GitHub: [@ayush123-bit](https://github.com/ayush123-bit)

## 📜 License

This project is licensed under the [MIT License](LICENSE).

---

> **Disclaimer**: This tool is intended for educational and demonstration purposes only. It should not be used for actual medical diagnosis.
