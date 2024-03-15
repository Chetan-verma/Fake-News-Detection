
import joblib
from sklearn.linear_model import LogisticRegression

# doc_new = ['obama is running for president in 2016']

var = input("Please enter the news text you want to verify: ")
print("You entered: " + str(var))

# function to run for prediction
def detecting_fake_news(var):
    # retrieving the best model for prediction call
    with open('final_model.sav', 'rb') as model_file:
        load_model = joblib.load(model_file)



        # Prediction and probability calculation
        prediction = load_model.predict([var])
        prob = load_model.predict_proba([var])

    return (
        print("The given statement is ", prediction[0]),
        print("The truth probability score is ", prob[0][1])
    )

detecting_fake_news(var)

