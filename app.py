from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load model and data
model = joblib.load('final_mlp_model.pkl')
features = joblib.load('feature_columns.pkl')

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None

    if request.method == 'POST':
        try:
            input_data = []
            for feature in features:
                value = request.form.get(feature)
                if value is None or value.strip() == '':
                    return render_template('index.html', features=features, result=None, error=f"Missing value for {feature}")
                input_data.append(float(value))

            input_array = np.array(input_data).reshape(1, -1)
            prediction = model.predict(input_array)[0]
            result = prediction

        except Exception as e:
            return render_template('index.html', features=features, result=None, error=str(e))

    return render_template('index.html', features=features, result=result)

if __name__ == '__main__':
    app.run(debug=True)
