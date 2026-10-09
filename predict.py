import joblib

# Load saved files
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("spam_model.pkl")

print("================================")
print("       SPAM DETECTION")
print("================================")

while True:

    message = input("\nEnter a message (type 'quit' to exit): ")

    if message.lower() == "quit":
        print("Goodbye!")
        break

    message_tfidf = vectorizer.transform([message])

    prediction = model.predict(message_tfidf)

    probability = model.predict_proba(message_tfidf)

    ham_probability = probability[0][0]
    spam_probability = probability[0][1]

    print("\nPrediction:", prediction[0].upper())

    if prediction[0] == "spam":
        print(f"Confidence: {spam_probability * 100:.2f}%")
    else:
        print(f"Confidence: {ham_probability * 100:.2f}%")