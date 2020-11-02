# Disaster Response Pipeline Project

This project is aimed at building an ETL and ML pipeline to understand the relevance of text messages received by emergency response centres. 

### Files relevant to this project.
Relevant files and directories: 
* app/run.py --> This executes the web front end.
* data/process_data.py --> This extracts the data from the flat files and loads it to a sqlite dB.
* models/train_classifier.py --> train_classifier builds and trains the model with an evaluation. 
* Flat data is stored in the data files disaster_categories.csv, and disaster_messages.csv

In order to run the application you will require the following Python repositories.
* nltk
* numpy
* pandas
* pickle
* plotly
* scikit-learn (a.k.a - sklearn)
* sqlalchemy

### How to run this project
You can find the detailed steps in the Project 3 readme. 
I have included the steps here as well, just in case. 

1. This will extract the data from the provided csv files and output a sqlite dB (run this from the project root).
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2. This step loads data from the dB, trains and evaluates the model. Once done it will write the model to a Pickle file. Step 2: This step loads data from the dB, trains and evaluates the model. Once done it will write the model to a Pickle file. (run this from the project root).   
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
3. This runs the web interface to show you the analysis of the model. (run this from the project app directory) 
`python run.py`
4. You can access the inter face from the below address.
`Go to http://0.0.0.0:3001/`
