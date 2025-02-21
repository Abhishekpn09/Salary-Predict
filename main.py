import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

from dataset.load import load_df
from sklearn.preprocessing import RobustScaler, StandardScaler, MaxAbsScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import KFold, GridSearchCV
from config.utils import evaluate
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score

import joblib


scoring = {'accuracy': 'accuracy',
           'precision': 'precision',
           'recall': 'recall'}

df = load_df()
print(f"Dataframe shape: {df.shape}")
print(df.head(2))

#print(df.info)
print()
print("*" * 25)
print(":: GET TRAIN=TEST SPLIT")
print("-"*25)

train = df.sample(frac=0.95, random_state=42)
test = df.drop(train.index)

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print("*" * 25)

X_train = train.drop(["Salary"], axis=1)
y_train = train["Salary"].values
X_test= test.drop(["Salary"], axis=1)
y_test = test["Salary"].values

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
print("*" * 25)
print("")



print(":: MODEL SELECTION ::")
print(":: 1. Decission Tree")

"""
# Define the column transformer
transform = ColumnTransformer([
    ("label", OrdinalEncoder(), ["EdLevel", "Country", "Age"]),
    ("onehot", OneHotEncoder(handle_unknown="ignore"), ["RemoteWork"]),
    ("scaler", MaxAbsScaler(), ["YearsCodePro"])
], remainder="passthrough")

# Define the model
model = DecisionTreeRegressor(random_state=42)

# Create the pipeline
pipe = Pipeline([
    ("preprocess", transform),
    ("model", model)
])

# Define the KFold cross-validator
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the scoring metrics
scoring = {
    'neg_root_mean_squared_error': 'neg_root_mean_squared_error',
    'neg_mean_absolute_error': 'neg_mean_absolute_error',
    'r2': 'r2'
}

# Perform cross-validation
scores = cross_validate(pipe, X_train, y_train, cv=kfold, scoring=scoring, n_jobs=2)

# Print the results
print(f"RMSE: mean: {-1 * np.mean(scores['test_neg_root_mean_squared_error'])} | values: {-1 * scores['test_neg_root_mean_squared_error']}")
print(f"MAE: mean: {-1 * np.mean(scores['test_neg_mean_absolute_error'])} | values: {-1 * scores['test_neg_mean_absolute_error']}")
print(f"R2-score: mean: {np.mean(scores['test_r2'])} | values: {scores['test_r2']}")
print("*" * 69)
print()
"""

print("Obtained Results...")
print("RMSE: mean: 55992.50961072083 | values: [56915.91968987 54922.62340903 56415.75092781 55626.81046918 56081.44355773]")
print("MAE: mean: 39782.97033741816 | values: [40217.44619775 39378.44821219 39887.73342286 39739.31299312 39691.91086117]")
print("R2-score: mean: 0.16243255543755966 | values: [0.14526938 0.18849746 0.13847371 0.17569686 0.16422537]")
print("*********************************************************************")

print(":: 2. AdaBoost")
print("-" * 25)

"""
transform = ColumnTransformer([
    ("label", OrdinalEncoder(), ["EdLevel", "Country", "Age"]),
    ("onehot", OneHotEncoder(handle_unknown="ignore"), ["RemoteWork"]),
    ("scaler", MaxAbsScaler(), ["YearsCodePro"])
], remainder="passthrough")

# Use 'estimator' instead of 'base_estimator'
model = AdaBoostRegressor(estimator=DecisionTreeRegressor(), n_estimators=200, random_state=42)

pipe = Pipeline([
    ("preprocess", transform),
    ("model", model)
])

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_validate(pipe, X_train, y_train, cv=kfold, scoring=scoring, n_jobs=2)
print(f"RMSE: mean: {-1 * np.mean(scores['test_neg_root_mean_squared_error'])} | {-1 * scores['test_neg_root_mean_squared_error']}")
print(f"MAE: mean: {-1 * np.mean(scores['test_neg_mean_absolute_error'])} | {-1 * scores['test_neg_mean_absolute_error']}")
print(f"R2-score: mean: {np.mean(scores['test_r2'])} | {scores['test_r2']}")
print("*" * 69)
print()
"""

print("Obtained Results...")
print("RMSE: mean: 38409.56149889353 | [39058.59223643 37560.25897727 39103.65603966 38209.28339411 38116.016847  ]")
print("MAE: mean: 26976.714230996102 | [27421.49343551 26297.32881856 27285.16157196 27104.96963013 26774.61769883]")
print("R2-score: mean: 0.6058102785900579 | [0.59747309 0.62047141 0.58609324 0.61108345 0.61393019]")
print("*********************************************************************")

print(":: 3. Bagging")
"""
transform = ColumnTransformer([
    ("label", OrdinalEncoder(), ["EdLevel", "Country", "Age"]),
    ("onehot", OneHotEncoder(handle_unknown="ignore"), ["RemoteWork"]),
    ("scaler", MaxAbsScaler(), ["YearsCodePro"])
], remainder="passthrough")

# Use 'estimator' instead of 'base_estimator'
model = BaggingRegressor(estimator=DecisionTreeRegressor(), n_estimators=200, n_jobs=2, random_state=42)

pipe = Pipeline([
    ("preprocess", transform),
    ("model", model)
])

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Define appropriate scoring metrics for regression
scoring = {
    'neg_root_mean_squared_error': 'neg_root_mean_squared_error',
    'neg_mean_absolute_error': 'neg_mean_absolute_error',
    'r2': 'r2'
}

scores = cross_validate(pipe, X_train, y_train, cv=kfold, scoring=scoring, n_jobs=2)

print(f"RMSE: mean: {-1 * np.mean(scores['test_neg_root_mean_squared_error'])} | {-1 * scores['test_neg_root_mean_squared_error']}")
print(f"MAE: mean: {-1 * np.mean(scores['test_neg_mean_absolute_error'])} | {-1 * scores['test_neg_mean_absolute_error']}")
print(f"R2-score: mean: {np.mean(scores['test_r2'])} | {scores['test_r2']}")
print("*" * 69)
print()
"""
print("Obtained Results...")
print("RMSE: mean: 38405.694654109524 | [39140.77138572 37591.1461938  39012.75834672 38133.41053562 38150.38680869]")
print("MAE: mean: 27321.283015342495 | [27756.46881837 26872.92638932 27679.76574077 27338.37372876 26958.8803995 ]")
print("R2-score: mean: 0.6058999642059488 | [0.59577748 0.61984696 0.58801528 0.61262648 0.61323363]")
print("*********************************************************************")

print(":: 4. RandomForest")
# Define the scoring metrics
scoring = {
    'neg_root_mean_squared_error': 'neg_root_mean_squared_error',
    'neg_mean_absolute_error': 'neg_mean_absolute_error',
    'r2': 'r2'
}

print(":: RandomForestRegressor")
print("-" * 25)
"""
transform = ColumnTransformer([
    ("label", OrdinalEncoder(), ["EdLevel", "Country", "Age"]),
    ("onehot", OneHotEncoder( handle_unknown="ignore"), ["RemoteWork"]),
    ("scaler", MaxAbsScaler(), ["YearsCodePro"])
], remainder="passthrough")

model = RandomForestRegressor(n_estimators=200, n_jobs=2, random_state=42)

pipe = Pipeline([
    ("preprocess", transform),
    ("model", model)
])

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_validate(pipe, X_train, y_train, cv=kfold, scoring=scoring, n_jobs=2)

print(f"RMSE: mean: {-1 * np.mean(scores['test_neg_root_mean_squared_error'])} | {-1 * scores['test_neg_root_mean_squared_error']}")
print(f"MAE: mean: {-1 * np.mean(scores['test_neg_mean_absolute_error'])} | {-1 * scores['test_neg_mean_absolute_error']}")
print(f"R2-score: mean: {np.mean(scores['test_r2'])} | {scores['test_r2']}")
print("*" * 69)
print()
"""
print("Obtained Results...")
print("RMSE: mean: 38405.35622318387 | [39143.79236694 37597.26426918 39014.35033854 38132.17677086 38139.1973704 ]")
print("AE: mean: 27328.064394861052 | [27755.72633691 26890.21596046 27689.27988697 27325.49225797 26979.60753199]")
print("R2-score: mean: 0.605906390476256 | [0.59571508 0.6197232  0.58798166 0.61265154 0.61346047]")
print("*********************************************************************")

print(":: 5. Gradiant Boost")

# Define the scoring metrics
scoring = {
    'neg_root_mean_squared_error': 'neg_root_mean_squared_error',
    'neg_mean_absolute_error': 'neg_mean_absolute_error',
    'r2': 'r2'
}

print(":: GradientBoostingRegressor")
print("-" * 25)
"""
transform = ColumnTransformer([
    ("label", OrdinalEncoder(), ["EdLevel", "Country", "Age"]),
    ("onehot", OneHotEncoder(handle_unknown="ignore"), ["RemoteWork"]),
    ("scaler", MaxAbsScaler(), ["YearsCodePro"])
], remainder="passthrough")

model = GradientBoostingRegressor(n_estimators=200)

pipe = Pipeline([
    ("preprocess", transform),
    ("model", model)
])

kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_validate(pipe, X_train, y_train, cv=kfold, scoring=scoring, n_jobs=2)

print(f"RMSE: mean: {-1 * np.mean(scores['test_neg_root_mean_squared_error'])} | {-1 * scores['test_neg_root_mean_squared_error']}")
print(f"MAE: mean: {-1 * np.mean(scores['test_neg_mean_absolute_error'])} | {-1 * scores['test_neg_mean_absolute_error']}")
print(f"R2-score: mean: {np.mean(scores['test_r2'])} | {scores['test_r2']}")
print("*" * 69)
print()

"""
print("Obtained Results...")
print("RMSE: mean: 37661.828244462995 | [38471.43762186 36724.84056721 38193.42348186 37439.75296082 37479.68659057]")
print("MAE: mean: 26609.23811899212 | [26979.60934633 26171.13484977 26834.41014172 26644.11471414 26416.92154301]")
print("R2-score: mean: 0.621018707110979 | [0.60948423 0.63716667 0.60513834 0.62659117 0.62671313]")
print("*********************************************************************")

"""
print(":: Hyperparameter Tuning...")
# Define the scoring metrics
scoring = {
    'neg_root_mean_squared_error': 'neg_root_mean_squared_error',
    'neg_mean_absolute_error': 'neg_mean_absolute_error',
    'r2': 'r2'
}

# Define the preprocessing and model
transform = ColumnTransformer([
    ("label", OrdinalEncoder(), ["EdLevel", "Country", "Age"]),
    ("onehot", OneHotEncoder(handle_unknown="ignore"), ["RemoteWork"]),
    ("scaler", MaxAbsScaler(), ["YearsCodePro"])
], remainder="passthrough")

model = GradientBoostingRegressor(random_state=42)

# Define the parameter grid
params = {
    "n_estimators": range(200, 510, 50),
    "loss": ['squared_error', 'absolute_error', 'huber', 'quantile'],
    "learning_rate": [0.01, 0.1, 0.2, 0.3, 0.4],
    "criterion": ['friedman_mse', 'squared_error']
}

# Setup GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=params, scoring=scoring, n_jobs=3, verbose=1, cv=3, refit="r2")

# Create the pipeline
pipe = Pipeline([
    ("preprocess", transform),
    ("grid", grid)
])

# Fit the pipeline
pipe.fit(X_train, y_train)

# Print the best parameters and best score for refit metric (r2)
print(f"The best params: {pipe['grid'].best_params_}")
print(f"The best r2 score: {pipe['grid'].best_score_}")

# Optional: Print best scores for each metric
best_index = pipe['grid'].best_index_
best_scores = pipe['grid'].cv_results_
print(f"Best RMSE: {-1 * best_scores['mean_test_neg_root_mean_squared_error'][best_index]}")
print(f"Best MAE: {-1 * best_scores['mean_test_neg_mean_absolute_error'][best_index]}")
print(f"Best R2: {best_scores['mean_test_r2'][best_index]}")
"""
print("The best params: {'criterion': 'friedman_mse', 'learning_rate': 0.1, 'loss': 'huber', 'n_estimators': 400}")
print("The best score: 0.6147934969813488")
print()

print(":: TRAIN AND SAVE BEST MODEL")
transform = ColumnTransformer([
    ("label", OrdinalEncoder(), ["EdLevel", "Country", "Age"]),
    ("onehot", OneHotEncoder( handle_unknown="ignore"), ["RemoteWork"]),
    ("scaler", MaxAbsScaler(), ["YearsCodePro"])
], remainder="passthrough")

model = GradientBoostingRegressor(criterion='friedman_mse', 
                                learning_rate=0.1, 
                                loss='huber', 
                                n_estimators= 400)

pipe = Pipeline([
    ("preprocess", transform),
    ("model", model)
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
evaluate(y_test, y_pred)

joblib.dump(pipe, "best_model.joblib")

