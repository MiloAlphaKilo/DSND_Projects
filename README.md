# DSND_Projects

This is a repo that will be used for ALL the projects required from them [Udacity Nano Degree in Data Science](https://www.udacity.com/course/data-scientist-nanodegree--nd025?utm_source=gsem_brand&utm_medium=ads_n&utm_campaign=8301633066_c&utm_term=85414326876&utm_keyword=udacity%20data%20science%20nanodegree_e&gclid=EAIaIQobChMIqq6P6Nug6wIVSOztCh3AvQGCEAAYAiAAEgJeifD_BwE).

## Project 3 - Disaster Response Pipeline

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

## Project 1 - Write a blog

### The Blog Post
The blog post can be found [here](https://lunkwillandfook.dev/2020/08/17/are-you-in-sales-or-hr-chances-are-you-might-be-leaving-your-job/).

### Motivation behind the project
The motivation behind this was to demonstrate the [CRISP Data modelling](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) process covered in the course. There are 3 notebook in the directory the repo does not cover stage 1 understanding the business problem, this will be covered in the commentary on the blog post. The notebooks will cover the topics of:

* Understand the data
* Clean the data
* Model the data

The output of the model will be used to then convey the message to stakeholders via a blog post.

### Libraries used

* numpy
* pandas
* matplotlib
* scikit-learn
* seaborn

### Files in the repo
The project was completely done in jupyter notebooks and you will probably need to have it running, unless there is something clever out there I don't know about - this is entirely possible.

* IBM_HR_Employee_Attrition.csv
* IBM_HR_Employee_Attrition_removed_data.csv
* 1_Understand_The_Data.ipynb
* 2_Data_Cleansing.ipynb
* 3_Build_the_Model.ipynb

#### IBM_HR_Employee_Attrition & IBM_HR_Employee_Attrition_removed_data
I obtained the IBM HR data through [Kaggle](https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset).
The quality of the data set is great and should be as it seems to have been generated by IBM. This did pose a challenge in the section where data cleaning is required. I copied the data set to the IBM_HR_Employee_Attrition_removed_data file and removed some data, randomly.

#### 1_Understand_The_Data.ipynb
This file looks simplistically at the data using pandas data-frames to get a general sense of the set.

#### 2_Data_Cleansing.ipynb
Here I tackle the problem of missing values in the data set to prep the data for the model. At each stage of clean up you will find a `df.isnull().sum()` line that will show the cleaning progress made.

#### 3_Build_the_Model.ipynb
The model is the meat and potatoes of the project where I cover a few steps to building the model. There is still some further cleanup here as there are columns that added very little value:
* StandardHours
* EmployeeCount
As well as dealing with the categorical values in the dataset.

The model creates some histograms of the data frame some are useful some aren't. There is also a seaborn heat map included to show some correlations between dimensions.

The first project's jupyter notebooks can be found in the Project 1 folder.

### Technicalities
There are a few points to note on the data set.

* The data set is fictional
* The data set is clean
* The data set feels on the small side
* There are a few biases in the data
  - It's weighted towards the male gender
  - It has a strong representation from the R & D department