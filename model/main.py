import pandas as pd
from LinearReg import CustomLinearRegression
from knn import knn
from decisiontree import decision_tree
from gradientboost import gradient_boost
from nueralnet import nueral_net
from randomforest import random_forest
from xgb import xgb_model


airport_to_location = {
    'ORD': (41.978611, -87.904724),
}
filepath = '../data/Awt.cbp.gov_ORD_2024-09-05-2025-09-05.csv'
data = pd.read_csv(filepath)
aircode = filepath[filepath.find('_') + 1:filepath.find('_') + 4]
train_set = data.sample(frac = 0.8, random_state = 0)
test_set = data.drop(train_set.index)

#CustomLinearRegression(train_set, test_set, airport_to_location)

#KNN
#knn(train_set, test_set, airport_to_location)

#decision tree
#decision_tree(train_set, test_set, airport_to_location)

#gradient boosting
#gradient_boost(train_set, test_set, airport_to_location, aircode)

xgb_model(train_set, test_set, airport_to_location, aircode)


#random forest0
#random_forest(train_set, test_set, airport_to_location)

#nueral network
#nueral_net(train_set, test_set, airport_to_location)



