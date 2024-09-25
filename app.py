from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from xgboost import XGBRegressor

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')
transformer = joblib.load('preprocessor.pkl')

def predict_fuel_price(day, month, year, city, fuel_type):
    data = {'city':[city],
            'fuel_type':[fuel_type],
            'month':[month],
            'year':[year],
            'day':[day]} 

    input_df = pd.DataFrame(data)
    input_df.head()
    transformed_data =transformer.transform(input_df)
    predicted_price = model.predict(transformed_data)
    return predicted_price[0]



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    day = data['day']
    month = data['month']
    year = data['year']
    city = data['city']
    fuel_type = data['fuelType']
    print(month,year,city,fuel_type)
    
    predicted_price = predict_fuel_price(day, month, year, city, fuel_type)
    
    return jsonify({'predicted_price': predicted_price})

if __name__ == '__main__':
    app.run(debug=True)