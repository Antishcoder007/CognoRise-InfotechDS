from flask import Flask, render_template,url_for,request
import joblib

model = joblib.load('./model/LogicalRegression.lb')

app = Flask(__name__)

@app.route('/')
def home():
    result = ''
    return render_template('ireshome.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == "POST":
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        prediction = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])[0]

        label_dict = {'0':'Iris-setosa','1':'Iris-versicolor','2':'Iris-virginica'}

        return render_template('ireshome.html',output = label_dict[str(prediction)])

if __name__ == '__main__':
    app.run(debug=True)