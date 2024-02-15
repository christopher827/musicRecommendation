import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
 
music_data = pd.read_csv('music.csv')  # Imports the dataset/model
X=music_data.drop(columns=['genre']) #Creates a new dataset without "genre" =>age&gender =>The input variable
y=music_data['genre'] #outputs the answers/music-type =>Genre=>The target variable

# Split the data into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2) #Allocating %20 of our data for testing and %80 for training

# Create a decision tree model
model=DecisionTreeClassifier() #Creates a model variable
model.fit(X_train,y_train) #Trains the data,learns the relationship between the input and output data
predictions=model.predict(X_test) #Test the untrained data in the model
score=accuracy_score(y_test,predictions) #How accurate the prediction(s) is/are
# print(score)
# print(predictions)
joblib.dump(model,'musicRecommender.joblib') #saves the already trained model
# model=joblib.dump('musicRecommender.joblib') loads the saved trained model
# predictions=model.predict([[21,1]])
#DecisionTreeClassifier=>Is a machine learning algorithm
#age,gender,genre

# music_data = pd.read_csv('music.csv')
# X=music_data.drop(columns=['genre']) 
# y=music_data['genre']    
# model.fit(X,y) 
# prediction=model.fit([ [21,1], [22,0] ])
#print(prediction)