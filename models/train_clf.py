import pandas as pd

sys.path.insert(0, "/Users/jhussman/Documents/predict-energy")
sys.path.insert(0, "/Users/jhussman/Documents/predict-energy/features")
from models import FeedForwardNN
from features import DataMatrix, load_data, TorchDataSet
from utils import load_fuel_data, nn_clf_labels


"""
File for training the classification model
------------------------------------------

Model details and results:
-3-layer Feed Forward Neural Network with:
    -Hidden Layer1 = 256 neurons
    -Hidden Layer2 = 256 neurons

FeedForward Neural Network: Validation Accuracy = 91.1% --- Test Accuracy = 90.4% --- Training Accuracy = 94.2%
Optimal Hyper-parameters:
-learning_rate = 0.0005
-betas = (0.9, 0.999)
-num_epochs=50

"""

# Path to lib
lib = '/Users/jhussman/Documents/predict-energy/lib/'

# Features dictionary defines the instructions for extracted feature data from raw data csv
feature_dict = {
    'load': {'path': lib + 'data/load_data.csv', 'col': 'Load'},
    'solar': {'path': lib + 'data/Solar_data.csv', 'col': 'Generation'},
    'wind': {'path': lib + 'data/Wind_data.csv', 'col': 'Generation'},
    'outages_hydro': {'path': lib + 'data/outages_data.csv', 'col': 'Outage', 'q': {'col': 'Type', 'rows': 'Hydro'}},
    'outages_thermal': {'path': lib + 'data/outages_data.csv', 'col': 'Outage', 'q': {'col': 'Type', 'rows': 'Thermal'}},
    'outages_vre': {'path': lib + 'data/outages_data.csv', 'col': 'Outage', 'q': {'col': 'Type', 'rows': 'Renewable'}}
}

# Initiate the DataMatrix object defined by the target vector
op = nn_clf_labels()
matrix = DataMatrix(op, target_labels=['LI_op'])

# Add features
matrix.add_fourier_terms()
matrix.add_previous_days(price=load_data(pd.read_csv(lib + 'data/lmp_data.csv', index_col='OPR_DT'), 'Price'))
matrix.add_features(feature_dict)
matrix.add_fuel_price(fuel_df=load_fuel_data())
matrix.add_clf()

# Initiate the Feed Forward Neural Network wrapper
n_nodes = [len(matrix.feature_names()), 256, 256]
model = FeedForwardNN(n_nodes, regression=False)

# Fit the MinMaxScaler to the Feature data
matrix.df[matrix.feature_names()] = model.fit_scaler(matrix.df[matrix.feature_names()])

# Initiate TorchDataSet object that converts DataMatrix to Pytorch usable format
torch_data = TorchDataSet(matrix)

# Split data into training set, validation set (for hyper-parameter tuning) and final testing set
train, validate, test = torch_data.process_splits(batch=128)

# Train model and score each set
score_train, score_validate = model.train_model(train, validate, learning_rate=0.0005, num_epochs=10, metric='accuracy')
score_test = model.test_model(test)

# Save the model and scaler as pickle files
model.save_model(lib + 'models/clf_ffnn.pt')
model.save_scaler(lib + 'models/clf_scaler.pkl')

