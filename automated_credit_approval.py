# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Load the dataset
cc_apps = pd.read_csv("cc_approvals.data", header=None) 
cc_apps.head()
cc_apps.info()
print(cc_apps.isin(["?", "NA", "N/A", "None"]).sum())

# Preprocess data
missing_markers = ["?", "NA", "N/A", "None", "", " "]
cc_apps = cc_apps.replace(missing_markers, np.nan)
cc_apps.isna().sum().sort_values(ascending=False)
cc_apps[0].isna().sum()

# Keep track of changes with df
cc_apps_adj = cc_apps 

for col in cc_apps_adj.columns:
    if cc_apps_adj[col].dtype == 'object':
        cc_apps_adj[col] = cc_apps_adj[col].fillna(cc_apps_adj[col].mode()[0])
    else:
        cc_apps_adj[col] = cc_apps_adj[col].fillna(cc_apps_adj[col].mean())
# Sanity check
cc_apps_adj.isna().sum()

# One-hot encoding
map = {"+": 1, "-": 0}
cc_apps_adj[13] = cc_apps_adj[13].map(map)
cc_apps_adj.head()
y = cc_apps_adj[13].values
cc_apps_adj = cc_apps_adj.drop(13, axis=1)
select_cols = cc_apps_adj.select_dtypes(include=["object"]).columns
cc_apps_adj_enc = pd.get_dummies(cc_apps_adj, columns=select_cols, drop_first=True)

# Prepare data for modelling
cc_apps_adj_enc.head()
X = cc_apps_adj_enc.values

# Train, test, split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Instantiate Logistic Regression and Standard Scaler -> Pipeline 
logreg = LogisticRegression(random_state=42)
scaler = StandardScaler()
steps = [('scaler', StandardScaler()),
        ('logreg', LogisticRegression())]
pipeline = Pipeline(steps)
pipeline.fit(X_train, y_train)
print(pipeline.score(X_test, y_test))
y_pred_pl = pipeline.predict(X_test)

# Confusion Matrix for pipeline model
print(confusion_matrix(y_test, y_pred_pl))

# params for hyperparameter tuning
params = {"logreg__solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
         "logreg__C": [0.01, 0.1, 1, 10]}

# Grid Search CV
logreg_cv = GridSearchCV(pipeline, param_grid=params, cv=5)
logreg_cv.fit(X_train, y_train)
test_score = logreg_cv.score(X_test, y_test)
y_pred_cv = logreg_cv.predict(X_test)
print(confusion_matrix(y_test, y_pred_cv))
print(test_score)

best_score = test_score
