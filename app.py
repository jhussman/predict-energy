from flask import Flask
from flask_restful import Api, reqparse, Resource
import numpy as np

from models import FeedForwardNN


"""

API for getting model predictions from forecast and classifier (clf) models
---------------------------------------------------------------------------

Forecast:

suffix = /forecast_lmp

Format:
params = {'hour': 1, 'day_of_week': 1, 'day_of_year': 23, 'price_minus_1': 23.45, 'load': 11295, 'solar': 7895.1, 
          'wind': 213.3, 'outages_hydro': 345., 'outages_thermal': 1675., 'outages_vre': 432.1, 'fuel_price': 4.22}

----------------------------------------------------------------------

Classifier:
suffix = /optimal_dispatch

Format:
params = {'hour': 1, 'day_of_week': 1, 'day_of_year': 23, 'price_minus_1': 23.45, 'load': 11295, 'solar': 7895.1, 
          'wind': 213.3, 'outages_hydro': 345., 'outages_thermal': 1675., 'outages_vre': 432.1, 'fuel_price': 4.22,
          'soc': 0.}

----------------------------------------------------------------------
Output:
{prediction: value, missing_data: [list of any missing parameters]}
value = Charge, Discharge or Idle

*Note that if there are any missing parameters, the model will assume a value of 0.

"""


app = Flask(__name__)
api = Api(app)

# Date arguments (converted to sine and cosine)
date_args = {'hour': 24., 'day_of_week': 7., 'day_of_year': 365.25}

# Other arguments, specific to forecasting model
forecast_args = ['price_minus_1', 'load', 'solar', 'wind', 'outages_hydro', 'outages_thermal', 'outages_vre',
                 'fuel_price']
num_inputs = len(forecast_args) + 2 * len(date_args)

# Load forecast and scaler models
model = FeedForwardNN([num_inputs, 128, 128])
model.load_model('lib/models/forecast_ffnn.pt')
model.load_scaler('lib/models/forecast_scaler.pkl')

# Create parser object and parse arguments
parser = reqparse.RequestParser()

for arg in date_args.keys():
    parser.add_argument(arg)

for arg in forecast_args:
    parser.add_argument(arg)


# Resource class for forecasting model
class ForecastPrice(Resource):
    def get(self):
        args = parser.parse_args()
        vector = list()
        missing_data = list()

        # Date time arguments converted to sine
        for arg_i, tau in date_args.items():
            val = args.get(arg_i, None)
            if val:
                vector.append(np.sin(2 * np.pi * float(val) / tau))
            else:
                vector.append(0)
                missing_data.append(arg_i)

        # Date time arguments converted to cosine
        for arg_i, tau in date_args.items():
            val = args.get(arg_i, None)

            if val:
                vector.append(np.cos(2 * np.pi * float(val) / tau))
            else:
                vector.append(1)
                missing_data.append(arg_i)

        # Forecast specific arguments
        for arg_i in forecast_args:
            val = args.get(arg_i, None)

            if val:
                vector.append(float(val))
            else:
                vector.append(0)
                missing_data.append(arg_i)

        vector = model.scaler.transform(np.array(vector).reshape(1, num_inputs))

        return {'prediction': round(float(model.predict(vector)[0][0]), 2), 'missing_data': missing_data}


"""
Classifier
 
"""

# Other arguments, specific to classifier model
clf_args = ['price_minus_1', 'load', 'solar', 'wind', 'outages_hydro', 'outages_thermal', 'outages_vre',
            'fuel_price', 'soc']
num_clf_inputs = len(clf_args) + 2 * len(date_args)

# Load classifier and scaler models
model_clf = FeedForwardNN([num_clf_inputs, 256, 256], regression=False)
model_clf.load_model('lib/models/clf_ffnn.pt')
model_clf.load_scaler('lib/models/clf_scaler.pkl')

# Create parser object and parse arguments
parser_clf = reqparse.RequestParser()

for arg in date_args.keys():
    parser_clf.add_argument(arg)

for arg in clf_args:
    parser_clf.add_argument(arg)


# Resource Class for classification model
class OptimalDispatch(Resource):
    def get(self):
        args = parser_clf.parse_args()
        vector = list()
        missing_data = list()

        # Date time arguments converted to sine
        for arg_i, tau in date_args.items():
            val = args.get(arg_i, None)

            if val:
                vector.append(np.sin(2 * np.pi * float(val) / tau))
            else:
                vector.append(0)
                missing_data.append(arg_i)

        # Date time arguments converted to cosine
        for arg_i, tau in date_args.items():
            val = args.get(arg_i, None)

            if val:
                vector.append(np.cos(2 * np.pi * float(val) / tau))
            else:
                vector.append(1)
                missing_data.append(arg_i)

        # Classifier specific arguments
        for arg_i in clf_args:
            val = args.get(arg_i, None)

            if val:
                vector.append(float(val))
            else:
                vector.append(0)
                missing_data.append(arg_i)

        # Scale vector and predict dispatch
        vector = model_clf.scaler.transform(np.array(vector).reshape(1, num_clf_inputs))
        category = np.argmax(model_clf.predict(vector))

        if category == 0:
            dispatch = 'Idle'
        elif category == 1:
            dispatch = 'Charge'
        else:
            dispatch = 'Discharge'

        return {'prediction': dispatch, 'missing_data': missing_data}


# Add the resources and their respective paths
api.add_resource(ForecastPrice, '/forecast_lmp')
api.add_resource(OptimalDispatch, '/optimal_dispatch')


if __name__ == '__main__':
    app.run(debug=True)


