import joblib

model=joblib.load('musicRecommender.joblib')
predictions=model.predict([[20,0]])
print(predictions)