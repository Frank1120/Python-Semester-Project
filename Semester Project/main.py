# Ideas:
# Create a machine learning prediction model that will take all of my Japanese Anki reviews and determine whether or not i will press "again" or "great" 
# Use logistic regression, analyzing multiple prior data points including days since review, previous successful reviews, previous failures, and the previous review interval 

import sqlite3
import pandas as pd
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

connection = sqlite3.connect('Semester Project/collection.anki2')

cursor = connection.cursor()

# Entire dataframe to allow reference to all columns 
full_dataframe = pd.read_sql_query("SELECT * FROM revlog JOIN cards ON revlog.cid = cards.id JOIN notes ON cards.nid = notes.id WHERE cards.did IN (1691725909438, 1691370644857, 1758915141730, 1758914389578)", connection)

ml_dataframe = pd.read_sql_query("SELECT revlog.id, revlog.cid, ease, revlog.ivl, lastivl, time, flds FROM revlog JOIN cards ON revlog.cid = cards.id " \
"JOIN notes ON cards.nid = notes.id " \
"WHERE cards.did IN (1691725909438, 1691370644857, 1758915141730, 1758914389578)", connection
                              )

ml_dataframe = ml_dataframe.sort_values(["cid", "id"])

ml_dataframe["id"] = ml_dataframe["id"] / 86400000 
ml_dataframe['Days Since Review'] = ml_dataframe.groupby("cid")["id"].diff() / 86400000

#print(full_dataframe)
print(ml_dataframe)

# Column ideas for multi variate 
# lastIvl
# ease? 
# currentIVL
# type maybe?
# Would want the revlog version of all of these, not card version
# Database resource https://github.com/ankidroid/Anki-Android/wiki/Database-Structure
# https://github.com/ankidroid/Anki-Android/wiki/Database%E2%80%90Structure%E2%80%902026#34-revlog--review-log
# Another database resource https://docs.ankiweb.net/manual/stats
# Leaning towards ivl, lastIvl and maybe one other

# Additional ideas on top of model prediction: Take all reviews and find an average amount of time per review, find average ivl, average time the review was at, etc etc. 
# Depending on how long cleaning data takes to figure out

connection.commit()
connection.close()