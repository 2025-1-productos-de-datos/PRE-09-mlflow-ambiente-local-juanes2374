import mlflow
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split

mlflow.start_run()

df = pd.read_csv("data/winequality-red.csv")
X = df.drop("quality", axis=1)
y = df["quality"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = ElasticNet()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

mlflow.log_metric("score", score)
mlflow.sklearn.log_model(model, "model")

mlflow.end_run()
