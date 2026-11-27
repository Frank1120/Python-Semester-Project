# Ideas:
# Create a machine learning prediction model that will take all of my Japanese Anki reviews and determine whether or not i will press "again" or "great" (i.e, get the review correct or not)
# Use logistic regression, analyzing multiple prior data points including days since review, previous successful reviews, previous failures, and the previous review interval 

import sqlite3
import pandas as pd
import numpy as np
import sklearn as sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

# Initial connection to allow sqlite manipulation 
connection = sqlite3.connect('Semester Project/collection.anki2')

# Entire dataframe to allow reference to all columns, testing purposes
full_dataframe = pd.read_sql_query('SELECT * FROM revlog JOIN cards ON revlog.cid = cards.id JOIN notes ON cards.nid = notes.id WHERE cards.did IN (1691725909438, 1691370644857, 1758915141730, 1758914389578)', connection)

# Actual machine learning dataframe that pulls particular columns needed to evaluate things like days since review, previous successes/failures, etc. 
# Removed flds while working with data columns to better visualize without unevenness
ml_dataframe = pd.read_sql_query('SELECT revlog.id, revlog.cid, ease, time, revlog.type FROM revlog JOIN cards ON revlog.cid = cards.id ' \
'JOIN notes ON cards.nid = notes.id ' \
'WHERE cards.did IN (1691725909438, 1691370644857, 1758915141730, 1758914389578) AND revlog.type < 4', connection
                              )

ml_dataframe = ml_dataframe.sort_values(['cid', 'id'])

# Creating 2 x column's for logistic regression, including adding a third called Successful to determine if the review was successful or not. 
ml_dataframe['Days Since Review'] = ml_dataframe.groupby('cid')['id'].diff() / 86400000
ml_dataframe['Successful'] = ml_dataframe['ease'] > 1
ml_dataframe['Previous Successes'] = ml_dataframe.groupby('cid')['Successful'].cumsum().groupby(ml_dataframe['cid']).shift().astype('Int64')
inverted = ~ml_dataframe['Successful']

# The below column was particularly tricky, needing to use groupby twice gave me trouble. Initially I was attempting to cumsum on a NaN value and receiving an error.
ml_dataframe['Previous Failures'] = inverted.groupby(ml_dataframe['cid']).cumsum().groupby(ml_dataframe['cid']).shift().astype('Int64')

# Creating a new dataframe with only type = 1, since in Anki, this signifies a regular scheduled review.
full_training_dataframe = ml_dataframe.query("type == 1")

# Creating x and y axis's with x representing only the data needed to train/make a prediction and y representing whether the respective training/prediction was correct/incorrect.
#column 7 9 and 10 i think 
X = full_training_dataframe[['Days Since Review', 'Previous Successes', 'Previous Failures']]
y = full_training_dataframe['Successful']

# This splits the original x and y lists into two separate lists for each variable, one for training the model and one for testing it.
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, shuffle=False)
model = make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', max_iter=1000))

# Training and fitting the model, using a custom prediction metric to adjust threshold.
model.fit(x_train, y_train)
custom_prediction = (model.predict_proba(x_test)[:, 0] <= 0.47).astype(int)
score = accuracy_score(y_test, custom_prediction)


# Total successful reviews in 20% test value
percent_true = y_test.mean()
#print(percent_true)

# Total predicted successful and unsuccessful reviews in 20% test value 
count = pd.Series(custom_prediction)
count = count.value_counts()

# The first column represents the odds of forgetting
print(count)


matrix1 = confusion_matrix(y_test, custom_prediction)
print(matrix1)

# Database resource https://github.com/ankidroid/Anki-Android/wiki/Database-Structure
# https://github.com/ankidroid/Anki-Android/wiki/Database%E2%80%90Structure%E2%80%902026#34-revlog--review-log
# Another database resource https://docs.ankiweb.net/manual/stats

# Additional ideas on top of model prediction: Take all reviews and find an average amount of time per review, find average ivl, average time the review was at, etc etc. 
# Depending on how long cleaning data takes to figure out

connection.commit()
connection.close()