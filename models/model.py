#Importing the necessary libraries
import pandas as pd
import pickle
#Classifier Libraries
from sklearn.ensemble import RandomForestClassifier


#Load the dataset
df = pd.read_csv('data/creditcard.csv')


#Selecting features for building model based upon the above density destribution plot
df = df[["Time",
         'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11','V12', 'V13', 'V14', 'V15', 'V16',
         'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28',
         "Class", 'Amount']]


#Training Data and Predictor Variable
x = df.drop('Class', axis = 'columns')
y = df['Class']


#Random Forest Model
regressor = RandomForestClassifier(n_estimators = 100, max_features = 3)
regressor.fit(x, y)


#Saving the model as pickle file
pickle.dump(regressor, open('model.pkl','wb'))


#Loading the model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 1]]))