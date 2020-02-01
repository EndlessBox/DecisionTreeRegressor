from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

# read data
iowa_file_path = "data_sets/train.csv"
home_data = pd.read_csv(iowa_file_path)

print(home_data.index)

# define the target_variable
# target_variable is the feature we will try to predict using our model.
y = home_data.SalePrice

# define and extract features
# we will use just some of the features:
features_list = ["LotArea", "YearBuilt",
                "1stFlrSF", "2ndFlrSF",
                "FullBath", "BedroomAbvGr",
                "TotRmsAbvGrd"]

# feature's list
# fearturs are the variable we will logic opon to make our prediction and find 
# pattern's !
X = home_data[features_list]


# split our dataset to validation and training part ! 
# to avoid the in_sample scoring.
# Sample scoring is when we train the model on the same data-test we will
# make our prediction on. 
# So, normaly we will get good scoring, but if we give him new 
# case's he will suck_dword .
# and for deep understanading check "DecisionTreeRegressor" man, and i know
# u smart_aword will get it.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

# DecisionTreeRegressor is a model that make predeiction based on tree out put.
# that mean : on each featur in the list he split the dataset into to groups
# and keep spliting group to 2^n . (2^n is max-leaf-size we can specifie it to know how much node 
# it will be between the start and the leaf of the tree.).
# playing with the "max-;eaf-size", can give us the optimum tree is size for our model.
hd_model = DecisionTreeRegressor()

# training our model on the already_splited dataset.
hd_model.fit(train_X, train_y)

# making prediction on the validation part of the dataset
predictions = hd_model.predict(val_X)

# Model validation : using MAE (Mean Absolute Error)
# Meam Absolute Error is when we calcul the deviation of all our error
# and we calculat the mean.
mea = mean_absolute_error(val_y, predictions)

print(mea)
