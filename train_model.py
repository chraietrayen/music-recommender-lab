import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

df = pd.read_csv("music.csv")
X = df.drop(columns=["genre"])
y = df["genre"]

model = DecisionTreeClassifier()
model.fit(X, y)

joblib.dump(model, "music_recommender.joblib")
print("Model saved!")
