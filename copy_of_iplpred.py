

import pandas as pd
import kagglehub

# Download latest version
path = kagglehub.dataset_download("adityabhaumik/ipl-2024-matches")

"""#**Load Dataset**

"""

df = pd.read_csv("/content/ipl2024 Matches.csv")



"""##**Data Preparation**

split data in x and y
"""

y = df["winner"]
x = df.drop("winner",axis = 1)

y

x

"""###Data Splitting"""

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=150)

x_train

x_test

"""# **Model Building**

##Linear Regression
"""

from sklearn.preprocessing import LabelEncoder
import random
df_copy = df.copy()

# Target and features
y = df_copy["winner"]
x = df_copy.drop("winner", axis=1)

# Encode the target (Label Encoding)
le_y = LabelEncoder()
y_encoded = le_y.fit_transform(y.astype(str))  # Make sure everything is string

# Encode the features (One-Hot Encoding for all object columns)
x_encoded = pd.get_dummies(x.astype(str))  # Convert all to string to avoid issues

# Split into train and test
x_train, x_test, y_train, y_test = train_test_split(x_encoded, y_encoded, test_size=0.2, random_state=2)

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
print("y_train length:", len(y_train))
print("y_test length:", len(y_test))

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train,y_train)

"""##**Appying the model to make prediction**"""

y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

decoded_train_preds = le_y.inverse_transform(y_lr_train_pred.astype(int))
decoded_train_preds

decoded_test_preds = le_y.inverse_transform(y_lr_train_pred.astype(int))
decoded_test_preds

from sklearn.metrics import mean_squared_error, r2_score

# Use the original encoded predictions and true labels
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)

print("lr_train_mse : ",lr_train_mse)
print("lr_train_r2 : ",lr_train_r2)
print("lr_test_mse : ",lr_test_mse)
print("lr_test_r2 : ",lr_test_r2)

lr_results = pd.DataFrame(["Linear Regression",lr_train_mse,lr_train_r2,lr_test_mse,lr_test_r2]).transpose()

lr_results.columns = ["Methods","Training MSE","Training R2","Testing MSE","Testing R2"]
lr_results

"""# **Random Forest**"""



"""Training the model"""

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=2,random_state=100)
rf.fit(x_train,y_train)

y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)

from sklearn.metrics import mean_squared_error, r2_score

# Use the original encoded predictions and true labels
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)

print("rf_train_mse",rf_train_mse)
print("rf_train_r2",rf_train_r2)
print("rf_test_mse",rf_test_mse)
print("rf_test_r2",rf_test_r2)

rf_results = pd.DataFrame(["Random Forest",rf_train_mse,rf_train_r2,rf_test_mse,rf_test_r2]).transpose()
rf_results.columns = ["Methods","Training MSE","Training R2","Testing MSE","Testing R2"]
rf_results

"""# **Model Comparision**"""

df_models = pd.concat([lr_results,rf_results],axis=0)
df_models

import matplotlib.pyplot as plt
plt.scatter(x = y_train,y=y_lr_train_pred,alpha=0.3)
plt.plot()

match_info = pd.DataFrame([{
    "team1": "Mumbai Indians",
    "team2": "Chennai Super Kings",
    "toss_winner": "Chennai Super Kings",
    "venue": "Wankhede Stadium",
    "season": 2024
}])
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

# For features
encoder = OrdinalEncoder()
x_encoded = encoder.fit_transform(x)

# For labels (winners)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# You MUST have these objects available
encoder = OrdinalEncoder()
encoder.fit(x)  # fit on training features

label_encoder = LabelEncoder()
label_encoder.fit(y)  # fit on training labels

print(x.columns.tolist())

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# 1. Assume x and y are your training features and target
# x = your training DataFrame
# y = your 'winner' column

# 2. Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 3. Convert training features using get_dummies
x_encoded = pd.get_dummies(x)

# 4. Fit the model
lr = LogisticRegression()
lr.fit(x_encoded, y_encoded)

# 5. Prepare new match info for prediction
match_info = pd.DataFrame([{
    "venue": "Ekana Cricket Stadium",
    "team1": "Lucknow",
    "team2": "Chennai",
    "stage": "Group Stage",
    "toss_winner": "Chennai",
    "toss_decision": "field",
    "date": "2025-04-14"
}])

# 6. Convert input match data to dummies
match_encoded = pd.get_dummies(match_info)

# 7. Align with training features (fill missing with 0)
match_encoded = match_encoded.reindex(columns=x_encoded.columns, fill_value=0)

# 8. Predict winner
pred = lr.predict(match_encoded)
predicted = label_encoder.inverse_transform(pred)[0]

# 9. Optional: Check if prediction is valid
team1 = match_info["team1"].values[0]
team2 = match_info["team2"].values[0]


proba = lr.predict_proba(match_encoded)[0]
labels = label_encoder.classes_
team1_idx = list(labels).index(team1)
team2_idx = list(labels).index(team2)

predicted = team1 if proba[team1_idx] > proba[team2_idx] else team2

print("✅ Predicted Winner:", predicted)