import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Training data
# Columns: study hours, sleep hours, classes attended
X = np.array([
    [1, 6, 3],
    [2, 7, 4],
    [3, 5, 5],
    [4, 8, 6],
    [5, 6, 7],
    [6, 7, 8],
    [2, 8, 3],
    [5, 5, 8],
    [3, 7, 4],
    [6, 8, 9]
], dtype=float)

# Labels: 0 = Fail, 1 = Pass
y = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

model.fit(X, y)

new_student = [[4, 7, 6]]

probability = model.predict_proba(new_student)[0, 1]
prediction = model.predict(new_student)[0]

print("Probability of passing:", round(probability, 3))
print("Prediction:", "Pass" if prediction == 1 else "Fail")
