from datasets import load_dataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle
import numpy
import sklearn

# Print library versions for debugging
print(f"Numpy version: {numpy.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")

# Load the dataset
dataset = load_dataset("legacy-datasets/banking77")

# Extract queries and labels
train_data = dataset['train']
train_queries = [example['text'] for example in train_data]
train_labels = [example['label'] for example in train_data]

# Map labels to intents
label_names = dataset['train'].features['label'].names
train_intent_labels = [label_names[label] for label in train_labels]

# Split data into training and validation
X_train, X_test, y_train, y_test = train_test_split(train_queries, train_intent_labels, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train Naive Bayes model
clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)

# Save model and vectorizer
with open("naive_bayes_model.pkl", "wb") as model_file:
    pickle.dump(clf, model_file)

with open("vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model and vectorizer saved successfully.")

# Confirm the files load correctly
with open("naive_bayes_model.pkl", "rb") as model_file:
    clf = pickle.load(model_file)
print("Model loaded successfully.")