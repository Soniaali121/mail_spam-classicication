📧 SpamShield AI — Email Spam Detection System

🚀 Overview

SpamShield AI is a machine learning-based email classification system that detects whether an email is:

🟢 Ham (Legitimate Email)
🔴 Spam (Unwanted Email)

It is built using Multinomial Naive Bayes, trained on a dataset of over 5,000 emails with 3,000+ word-frequency features.

This project demonstrates a full end-to-end ML pipeline for text classification.

🎯 Key Features

✔ High-performance spam detection (95%+ accuracy)
✔ Handles high-dimensional text data (3000+ features)
✔ Fast inference using Naive Bayes
✔ Interpretable results via confusion matrix
✔ Built using industry-standard ML workflow

🧠 Tech Stack
Python 🐍
Pandas / NumPy
Scikit-learn 🤖
Matplotlib / Seaborn 📊
📊 Dataset Overview
Property	Value
Total Emails	5172
Features	3000+ word frequencies
Classes	Spam / Ham
Imbalance	Yes (Ham: 71%, Spam: 29%)
🏗️ System Architecture
Email Dataset
      ↓
Data Preprocessing (Word Frequency Matrix)
      ↓
Train-Test Split (80/20)
      ↓
Multinomial Naive Bayes Model
      ↓
Prediction (Spam / Ham)
      ↓
Evaluation Metrics
⚙️ Machine Learning Pipeline
1. Load Dataset
import pandas as pd
spam_df = pd.read_csv("emails.csv")
2. Feature Selection
X = spam_df.iloc[:, 1:-1]
y = spam_df["Prediction"]
3. Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
4. Model Training
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB()
model.fit(X_train, y_train)
5. Prediction
y_pred = model.predict(X_test)
📈 Model Performance
🎯 Accuracy

95.45%

📊 Confusion Matrix
[[704  35]
 [ 12 284]]
📋 Classification Report
Class	Precision	Recall	F1-score
Ham (0)	0.98	0.95	0.97
Spam (1)	0.89	0.96	0.92
📉 Insights
Strong performance on both classes
High recall for spam detection (important for real-world use)
Naive Bayes works extremely well for word-frequency data
📊 Visualizations
📌 Spam vs Ham Distribution
Clear class imbalance (~70/30 split)
📌 Confusion Matrix Heatmap
Shows strong classification performance
🔮 Future Improvements (FAANG-Level Upgrades)
🔥 Replace Count Features with TF-IDF
🔥 Try Logistic Regression + SVM benchmarking
🔥 Add BERT-based deep learning model
🔥 Build REST API using Flask/FastAPI
🔥 Deploy using Streamlit / Docker / AWS
🔥 Add real-time email prediction UI
📦 Project Structure (Recommended)
spamshield-ai/
│
├── data/
│   └── emails.csv
│
├── notebooks/
│   └── spam_analysis.ipynb
│
├── src/
│   ├── train_model.py
│   ├── predict.py
│   └── preprocess.py
│
├── app.py
├── requirements.txt
└── README.md
🚀 How to Run
git clone https://github.com/your-username/spamshield-ai.git
cd spamshield-ai
pip install -r requirements.txt
python app.py
👩‍💻 Author

Sonia Ali
MSc Artificial Intelligence (UK)

⭐ If you like this project

Give it a ⭐ and connect on LinkedIn 🚀
