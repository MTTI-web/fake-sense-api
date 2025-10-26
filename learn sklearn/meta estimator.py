# Meta estimators are extra post-processing steps that are applied to the data after the pipeline is done

# %% importing modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=2000,
    n_features=2,
    n_redundant=0,
    random_state=21,
    class_sep=1.75,
    flip_y=0.1,
)

print(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=5)

# %% trying a linear regression
model = LogisticRegression()
model.fit(X, y)
pred = model.predict(X)
print("predictions:", pred)
plt.scatter(X[:, 0], X[:, 1], c=pred, s=5)

# %% running a voting classifier
clf1 = LogisticRegression().fit(X, y)
clf2 = KNeighborsClassifier(n_neighbors=10).fit(X, y)

# soft voting means we are averaging the predict_proba values
clf3 = VotingClassifier(
    estimators=[("clf1", clf1), ("clf2", clf2)], voting="soft", weights=[0.5, 0.5]
)
clf3.fit(X, y)

makePlot
