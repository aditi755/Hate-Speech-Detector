from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl','rb'))

@app.route("/")
def hello_world():
  return render_template('index.html')

@app.route('/api',methods=['POST'])
def predict():
    # Get the data from the POST request.
    data = request.get_json(force=True)
    # Make prediction using model loaded from disk as per the data.
    prediction = model.predict([[np.array(data['exp'])]])
    # Take the first value of prediction
    output = prediction[0]
    return jsonify(output)
if __name__ == '__main__':
    app.run(port=5000, debug=True)


if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True)

