#import libraries
import numpy as np
from flask import Flask, request, render_template
import pickle

#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    # Map the prediction to labels
    output_label = "Fraud Transaction" if prediction[0] == 1 else "Normal Transaction"

    return render_template('index.html', prediction_text=f'The transaction is: {output_label}')

if __name__ == "__main__":
    app.run(debug=True)