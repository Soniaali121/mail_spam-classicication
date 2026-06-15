# 📧 SpamShield AI — Email Spam Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Accuracy-95.45%25-green.svg" alt="Accuracy">
</p>

## 🚀 Overview
**SpamShield AI** is a machine learning-based email classification system designed to instantly flag unwanted messages. The system detects whether an email is:
* 🟢 **Ham** (Legitimate Email)
* 🔴 **Spam** (Unwanted Email)

Built using a **Multinomial Naive Bayes** classifier, it is trained on a dataset of over 5,000 emails processing 3,000+ distinct word-frequency features. This project showcases a clean, end-to-end machine learning pipeline optimized for text classification.

---

## 🎯 Key Features
* ✔ **High-Performance Text Classification:** Achieves a **95.45%** baseline accuracy.
* ✔ **High-Dimensional Data Handling:** Seamlessly processes 3,000+ sparse word frequencies.
* ✔ **Ultra-Fast Inference:** Utilizes Naive Bayes for near-instantaneous execution.
* ✔ **Production-Ready Layout:** Organized according to industry-standard modular project structures.

---

## 🧠 Tech Stack
* **Language:** Python 🐍
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn 🤖
* **Visualization:** Matplotlib, Seaborn 📊

---

## 📊 Dataset Overview

| Property | Value |
| :--- | :--- |
| **Total Emails** | 5,172 |
| **Features Included** | 3,000+ word frequencies |
| **Target Classes** | Spam / Ham |
| **Class Distribution** | Imbalanced (Ham: 71%, Spam: 29%) |

---

## 🏗️ System Architecture

```text
       [ Raw Email Dataset ]
                 │
                 ▼
    [ Data Preprocessing Pipeline ] -> (Word Frequency Matrix Extraction)
                 │
                 ▼
       [ Train-Test Split ] --------> (80% Train / 20% Test)
                 │
                 ▼
   [ Multinomial Naive Bayes Model ]
                 │
                 ▼
        [ Prediction Engine ] -------> (Classifies as Spam or Ham)
                 │
                 ▼
       [ Performance Metrics ] ------> (Accuracy, Precision, Recall, F1)

##⚙️ Machine Learning Pipeline
import pandas as pd

# Load processed tokenized email dataset
spam_df = pd.read_csv("data/emails.csv")

2. Feature Selection
# Separate frequency features from the label column
X = spam_df.iloc[:, 1:-1]
y = spam_df["Prediction"]


3. Train-Test Split
from sklearn.model_selection import train_test_split

# Split data keeping reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

##4. Model Training
from sklearn.naive_bayes import MultinomialNB

# Fit model on text frequencies
model = MultinomialNB()
model.fit(X_train, y_train)

##5. Prediction
# Evaluate predictions against test set
y_pred = model.predict(X_test)

##📈 Model Performance
##🎯 Accuracy

95.45%

*📊 Confusion Matrix
*[[704  35]
* [ 12 284]]

  
##📋 Classification Report
[[704  35]   <-- True Ham (704), False Spam (35)
 [ 12 284]]  <-- False Ham (12), True Spam (284)

###📉 Insights
Strong performance on both classes
High recall for spam detection (important for real-world use)
Naive Bayes works extremely well for word-frequency data

*📊 Visualizations
*📌 Spam vs Ham Distribution
*Clear class imbalance (~70/30 split)
*📌 Confusion Matrix Heatmap

###Shows strong classification performance
*🔮 Future Improvements (FAANG-Level Upgrades)
*🔥 Replace Count Features with TF-IDF
*🔥 Try Logistic Regression + SVM benchmarking
*🔥 Add BERT-based deep learning model
*🔥 Build REST API using Flask/FastAPI
*🔥 Deploy using Streamlit / Docker / AWS
*🔥 Add real-time email prediction UI
*📦 Project Structure (Recommended)


##📦 Project Structure

spamshield-ai/
│
├── data/
│   └── emails.csv               # Raw and processed datasets
│
├── notebooks/
│   └── spam_analysis.ipynb      # EDA and experimental model prototyping
│
├── src/
│   ├── train_model.py           # Core training execution script
│   ├── predict.py               # Evaluation script for inference
│   └── preprocess.py            # Custom text preprocessing functions
│
├── app.py                       # Main application entry point
├── requirements.txt             # Project dependencies
└── README.md                    # Documentation

##🚀 How to Run
1. Clone the repository:
Bash
   git clone [https://github.com/your-username/spamshield-ai.git](https://github.com/your-username/spamshield-ai.git)
   cd spamshield-ai

2.Set up a virtual environment & install requirements:

Bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt

3.Run the script:
   Bash
   python app.py
👩‍💻 Author
Sonia Ali
Give a ⭐ if you found this project helpful!

### 💡 Tips for Final Touches:
1. Replace `YOUR_LINKEDIN_URL_HERE` at the bottom with your actual LinkedIn profile link.
2. Replace `your-username` in the GitHub clone URL and profile link with your actual GitHub username.
