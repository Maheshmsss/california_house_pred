from flask import Flask, request, render_template
import numpy as np
import joblib

obj = joblib.load('california.joblib')

model = obj['Model']
columns = obj['Columns']

print(columns)

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html', columns=columns)


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    INPUT = []

    for i in columns:
        val = request.form.get(i)
        INPUT.append(float(val))

    prediction = model.predict([INPUT])

    return render_template(
        'index.html',
        columns=columns,
        prediction=prediction[0]
    )


if __name__ == '__main__':
    app.run(debug=True)