# predict-energy

This project contains various models for forecasting electricity prices and predicting the optimal dispatch of energy storage in the California Independent System Operator market. Included in the project is the raw data, trained models, scripts for extracting the features, defining the models, training the models as well as the code for the API.

/app - Contains the code for the API:
-------
	/Dockerfile - the Docker file for creating a Docker image of the app
	/Dockerrun.aws.json - the instructions for deploying the API to AWS Elastic Beanstalk (allows AWS to source the Docker image)
	/app.py - contains the Flask restful api resources
	/requirements.txt - app requirements
	/wsgi.py - gunicorn server 

/data - Contains the scripts for accessing the raw data from the CAISO API
-------
	/caiso_wrapper.py - class that wraps around the CAISO API

/features - Contains the scripts for extracting the featurues
-------
	/build_features.py - class that constructs the features from raw data
	/feature_schema.txt - outlines the features used in training the models

/lib - Library of raw data and trained models
-------
	/models - serialized models
	/data - csv files of raw data

/models - Model classes and scripts for building the models
-------
	/models.py - classes of different model types
	/train_clf.py - script to build, validate, train, test and save classification model
	/train_forecast.py - script to build, validate, train, test and save forecastin model


Contact j.hussman@columbia.edu for questions

