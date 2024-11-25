import pickle
import numpy
import sklearn

print(f"Numpy version: {numpy.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")

with open("naive_bayes_model.pkl", "rb") as model_file:
    clf = pickle.load(model_file)
print("Model loaded successfully.")
