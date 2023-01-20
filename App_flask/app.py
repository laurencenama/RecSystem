from flask import Flask, request, Response, render_template, jsonify
import os
import requests

# Initialize the Flask application
app = Flask(__name__)

# route http posts to this method

server = 'https://recsysp102.azurewebsites.net/api/httptrigger1'

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    user_id = request.form.get('user_id')
    response = requests.get(server+f'?user_id={user_id}')
    print(response)
    return render_template('predict.html', prediction=response.text)


# start flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
