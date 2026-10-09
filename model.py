import pandas as pd
import csv
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# -----------------------------
# 1. Load the dataset
# -----------------------------

rows = []

with open("SMSSpamCollection", "r", encoding="latin-1") as file:
    reader = csv.reader(file, delimiter="\t")

    for row in reader:
        if len(row) >= 2:
            label = row[0]
            message = "\t".join(row[1:])
            rows.append([label, message])

data = pd.DataFrame(rows, columns=["label", "message"])

# Remove duplicate messages
data = data.drop_duplicates()

print("Dataset shape:", data.shape)
print("\nLabel counts:")
print(data["label"].value_counts())


# -----------------------------
# 2. Separate input and output
# -----------------------------

X = data["message"]
y = data["label"]


# -----------------------------
# 3. Split the dataset
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))


# -----------------------------
# 4. Convert text into numbers
# -----------------------------
vectorizer = TfidfVectorizer(
    min_df=2,
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTraining data shape:", X_train_tfidf.shape)
print("Testing data shape:", X_test_tfidf.shape)


# -----------------------------
# 5. Train the model
# -----------------------------
model = MultinomialNB()


model.fit(X_train_tfidf, y_train)

print("\nModel training completed!")


# -----------------------------
# 6. Save the model and vectorizer
# -----------------------------

joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model saved as spam_model.pkl")
print("Vectorizer saved as tfidf_vectorizer.pkl")


# -----------------------------
# 7. Make predictions
# -----------------------------

prediction = model.predict(X_test_tfidf)
# -----------------------------
# 11. Test different thresholds
# -----------------------------

probabilities = model.predict_proba(X_test_tfidf)

# Column 1 = spam probability
spam_probabilities = probabilities[:, 1]

for threshold in [0.5, 0.4, 0.3, 0.2]:

    threshold_prediction = [
        "spam" if p >= threshold else "ham"
        for p in spam_probabilities
    ]

    precision = precision_score(
        y_test,
        threshold_prediction,
        pos_label="spam",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        threshold_prediction,
        pos_label="spam",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        threshold_prediction,
        pos_label="spam",
        zero_division=0
    )

    print(f"\nThreshold: {threshold}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1:.3f}")

# -----------------------------
# 8. Evaluate the model
# -----------------------------

accuracy = accuracy_score(y_test, prediction)

precision = precision_score(
    y_test,
    prediction,
    pos_label="spam",
    zero_division=0
)

recall = recall_score(
    y_test,
    prediction,
    pos_label="spam",
    zero_division=0
)

f1 = f1_score(
    y_test,
    prediction,
    pos_label="spam",
    zero_division=0
)

print("\n========== MODEL RESULTS ==========")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# -----------------------------
# 9. Confusion matrix
# -----------------------------

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, prediction))


# -----------------------------
# 10. Classification report
# -----------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        prediction,
        zero_division=0
    )
)
# -----------------------------
# 12. Error Analysis
# -----------------------------

results = pd.DataFrame({
    "message": X_test,
    "actual": y_test,
    "predicted": prediction
})

# Messages that were actually spam but predicted as ham
missed_spam = results[
    (results["actual"] == "spam") &
    (results["predicted"] == "ham")
]

# Messages that were actually ham but predicted as spam
false_spam = results[
    (results["actual"] == "ham") &
    (results["predicted"] == "spam")
]

print("\n========== ERROR ANALYSIS ==========")

print("\nMissed Spam Messages:")
print(missed_spam[["message"]].head(10))

print("\nHam Messages Incorrectly Classified as Spam:")
print(false_spam[["message"]].head(10))