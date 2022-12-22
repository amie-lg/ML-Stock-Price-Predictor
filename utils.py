import joblib
import numpy as np

def preprocess(Open, High, Low, Volume):
    test_data = np.array([[Open, High, Low, Volume]]) #loading in the inputs
    trained_model = joblib.load("model.plk") #loading in the model
    prediction = trained_model.predict(test_data) #prediction on the test data

    return prediction