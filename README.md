# Spam Detection using Machine Learning

## 📌 About the Project

This project is a Machine Learning based Spam Detection system that classifies SMS messages as **Spam** or **Ham (Not Spam)**.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- TF-IDF
- Multinomial Naive Bayes
- Joblib

## 🔄 Project Workflow

```text
SMS Dataset
    ↓
Data Cleaning
    ↓
Train/Test Split
    ↓
TF-IDF
    ↓
Naive Bayes Model
    ↓
Prediction
```

## 📊 Dataset

The project uses the **SMS Spam Collection** dataset.

After removing duplicate messages, **5,169 messages** were used.

## 📈 Model Performance

- Accuracy: **97.58%**
- Precision: **100%**
- Recall: **80.92%**
- F1 Score: **89.45%**

## 📁 Files

```text
model.py       → Train and evaluate the model
predict.py     → Predict new messages
requirements.txt → Required libraries
.gitignore     → Files ignored by Git
```

## ▶️ How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python model.py
```

Run the spam detector:

```bash
python predict.py
```

## 🎯 Example

```text
Input: You won a free prize

Prediction: SPAM
```

```text
Input: Can you call me tomorrow?

Prediction: HAM
```

## 📚 Learning Outcome

This project helped me understand **Python, NLP, TF-IDF, Machine Learning, Naive Bayes, model evaluation, and text classification**.