# udacity_data_scientist_track_project_2
NLP Natural Disaster Messaging Analysis

## Project Components
There are three components you'll need to complete for this project.

1. ETL Pipeline
In a Python script, process_data.py, write a data cleaning pipeline that:

* Loads the messages and categories datasets
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database

2. ML Pipeline
In a Python script, train_classifier.py, write a machine learning pipeline that:

* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file

3. Flask Web App
We are providing much of the flask web app for you, but feel free to add extra features depending on your knowledge of flask, html, css and javascript. For this part, you'll need to:

* Modify file paths for database and model as needed
* Add data visualizations using Plotly in the web app. One example is provided for you
* Github and Code Quality

Your project will also be graded based on the following:

* Use of Git and Github
* Strong documentation
* Clean and modular code

Code to run
```

python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

python train_classifier.py ../data/DisasterResponse.db classifier.pkl
```


## Code
The coding for this project can be completed using the Project Workspace IDE provided. Here's the file structure of the project:

```

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model
- README.md

```
