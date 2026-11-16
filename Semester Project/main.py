# Ideas:
# Create a machine learning prediction model that will take all of my Japanese Anki reviews and determine whether or not i will press "again" or "great" 
# Use logistic regression, analyzing multiple prior data points including days since review, previous successful reviews, previous failures, and the previous review interval 

import sqlite3
import pandas as pd

connection = sqlite3.connect('collection.anki2')

cursor = connection.cursor()

# cursor.execute("""CREATE TABLE IF NOT EXISTS reviews (
#         integer text,
#         cid integer,
#         ease integer,
#         ivl integer,
#         lastlvl integer,
#         factor real,
#         time real,
#         type integer
#         )""")

full_dataframe = pd.read_sql_query("SELECT * FROM revlog JOIN cards ON revlog.cid = cards.id JOIN notes ON cards.nid = notes.id WHERE cards.did IN (1691725909438, 1691370644857, 1758915141730, 1758914389578)", connection)

dataframe = pd.read_sql_query("SELECT revlog.id, revlog.cid, ease, revlog.ivl, lastivl, time, flds FROM revlog JOIN cards ON revlog.cid = cards.id " \
"JOIN notes ON cards.nid = notes.id " \
"WHERE cards.did IN (1691725909438, 1691370644857, 1758915141730, 1758914389578)", connection
                              )

print(full_dataframe)
#print(dataframe)



connection.commit()
connection.close()