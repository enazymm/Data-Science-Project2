# Disaster Response Project

## Introduction
This project is the second project in the Data Scientist Nanodegree by Udacity. The goal of this project is to use the concepts of ETL and machine learning pipelines to create a flask wep app to predict messages using existing data and classification.

## Documantation
The files provided in this project include:
1. **App Folder**: Include the template of the wep app and the run.py file to luanch the web app.
2. **Data Folder**: Include the CSV files that used in the training and the process_data.py which is used to clean and prepare the data in the database to used in the machine learning model.
3. **Models Folder**: Include the train_classifier.py that trains the model that will be used in the web app.

## How To Run The Code
After cloning the repository run the following commaned in the cloned directory
> python data/process_data.py data/disaster_categories.csv data/disaster_messages.csv data/DisasterResponse.db

This code will run the process_data.py file on the provided CSV files and store the cleaned data in the DisasterResponse database.

After that run the code
> python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

This will create the model that will be used in the web app

Finally run the following code inside the app folder
> python run.py

now you can access the web app using the followin link [Here](https://view6914b2f4-3001.udacity-student-workspaces.com/).

The web app looks like this

![Image1](https://i.imgur.com/crTfq6t.png)

It includes a input box for the message you want to analyze, the distribution of the genre and categories.

Once a message is entered the rusults look like this

![Image1](https://i.imgur.com/OhN5yeX.png)
