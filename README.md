# udacity_data_scientist_track_project_2
NLP Natural Disaster Messaging Analysis

## Backgrounds
When natural disaster happens, we would see millions of twitter messages regarding all sorts of topics around the disaster. There are import information that may help to save lives faster. Thus this web app developed to use machine learning to categorize twitter messages into multiple topics so that we could analyze in each category what is the most relevant information.

Project is from Udacity Data Scientist track.

## Results
In the web app we have included interactive link and several basic information:
* A message classifier on top to classify the twitter message based on the model we trained in the historical data. It can output one ore more of the 36 categories of the message.

* Web app graph 1: Interactive classifier and message genre overview ![Overview of genre](/Pictures/overview_of_genres.png)
* Web app graph 2: Pareto chart of most twitted topics ![Count of Topics ](/Pictures/count_of_topics.png)
* Web app graph 3: Area chart of message that having touched most topics ![Count of Topics ](/Pictures/messages_cover_most_topics.png)

## Project Components
There are three components included in this project.

1. ETL Pipeline
In a Python script, process_data.py, we have written a data cleaning pipeline that:

* Loads the messages and categories datasets
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database

2. ML Pipeline
In a Python script, train_classifier.py, we have writtten a machine learning pipeline that:

* Loads data from the SQLite database
* Splits the dataset into training and test sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs results on the test set
* Exports the final model as a pickle file

3. Flask Web App
Web app that having files to display results:

* Modify file paths for database and model as needed
* Add data visualizations using Plotly in the web app. One example is provided for you
* Github and Code Quality


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
