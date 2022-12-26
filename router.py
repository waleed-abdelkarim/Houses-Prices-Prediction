import numpy as np
import pandas as pd
import joblib
import os

import plotly.express as px
import plotly
import json

from flask import Flask, render_template, request, jsonify, make_response


from utils import preprocess_new


## initiliaze
app = Flask(__name__)


model = joblib.load('model_XGBoost.pkl')


@app.route('/')
@app.route('/home')
def home():
    FILE_PATH = os.path.join(os.getcwd(), 'housing.csv')
    df_housing = pd.read_csv(FILE_PATH)
  
    fig = px.scatter_mapbox(df_housing, 
                            lat="latitude", 
                            lon="longitude", 
                            hover_data=["population", "ocean_proximity"],
                            color="median_house_value",
                            color_continuous_scale=px.colors.sequential.Jet,
                            size=df_housing['population']/100,
    
                            zoom=4)

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)  
    return render_template('index.html', graphJSON=graphJSON)




@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST': 
        longitude = float(request.form['longitude'])
        latitude = float(request.form['latitude'])
        medianAge = float(request.form['medianAge'])
        totalRooms = float(request.form['totalRooms'])
        totalbedrooms = float(request.form['bedRooms'])

        population = float(request.form['population'])
        households = float(request.form['household'])
        medianIncome = float(request.form['medianIncome'])
        oceanProximity = request.form['oceanProximity']

        # Concatenate all Inputs
        X_new = pd.DataFrame({'longitude': [longitude], 'latitude': [latitude], 'housing_median_age': [medianAge], 'total_rooms': [totalRooms],
                              'total_bedrooms': [totalbedrooms], 'population': [population], 'households': [households], 'median_income': [medianIncome],
                              'ocean_proximity': [oceanProximity]
                              })

        # Feature Engineering we did
        X_new['rooms_per_household'] = X_new['total_rooms'] / X_new['households']
        X_new['bedrooms_per_rooms'] = X_new['total_bedrooms'] / X_new['total_rooms']
        X_new['pioulation_per_household'] = X_new['population'] / X_new['households']


        # Call the Function and Preprocess the New Instances
        X_processed = preprocess_new(X_new)

        # call the Model and predict
        y_pred_new = model.predict(X_processed)
        y_pred_new = '{:.3f}'.format(y_pred_new[0])

        # return the prediction as a json
        return render_template('predict.html', pred_val=y_pred_new, medianAge=medianAge, totalRooms=totalRooms, totalbedrooms=totalbedrooms, population=population, households=households, medianIncome=medianIncome, oceanProximity=oceanProximity, longitude=longitude, latitude=latitude)
    else:
        return render_template('predict.html')
 



## terminal
if __name__ == '__main__':
    app.run(debug=True)
