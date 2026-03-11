from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load data
model = joblib.load('best_model.pkl')
e_data = joblib.load('e_data.pkl')


@app.route('/')
def home():
    # Pass countries and continents to the HTML for dropdowns
    return render_template('home.html',
                           countries=e_data['countries'],
                           continents=e_data['continents'])


@app.route('/predict', methods=['POST'])
def predict():
    # Get data
    input_data = {
        'country': request.form.get('country'),
        'beer_servings': float(request.form.get('beer')),
        'spirit_servings': float(request.form.get('spirit')),
        'wine_servings': float(request.form.get('wine')),
        'continent': request.form.get('continent')
    }

    # Convert to DataFrame (Pipeline handles encoding/scaling)
    query_df = pd.DataFrame([input_data])
    prediction = model.predict(query_df)[0]

    return render_template('res.html',
                           prediction=round(prediction, 2),
                           countries=e_data['countries'],
                           continents=e_data['continents'])


if __name__ == '__main__':
    app.run(debug=True)
