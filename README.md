<div align="center">

  <img src="https://github.com/C23-PR571-Greenify/Greenify-Documentation/blob/main/logo.png" alt="logo" width="350" height="auto" />
  <h1>Greenify Machine Learning</h1>

</div>

Bangkit Capstone Team ID : C23 - PR571 <br>
Here is our repository for Bangkit 2023 Capstone project 

## Description
The idea of our application is to provide recommendations for environmentally friendly tourist attractions in the categories of culture, nature reserves and marine based on what users like and equipped with a description of tourist attractions, cities, prices and coordinates that can be accessed through Google Maps.

## Content-Based Recommendation System
A recommendation system is a technology used to provide suggestions or recommendations to users based on their preferences, behaviors, or gathered information about them. The main goal of a recommendation system is to assist users in discovering relevant and interesting content or products.

This system analyzes the existing content or attributes of products, such as descriptions, categories, or rating, and matches user preferences with similar content. For example, if a user shows an interest in tourist attractions with cultural category, the system will recommend other tourist attractions with cultural category that are similar.

## Dataset
Our dataset can be accessed [here](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination).

## Method
Since the current model is saved in HDF5 format, you can load the model in Python first:

```python
import tensorflow as tf

model = tf.keras.models.load_model('path_to_model/model.h5')
```

To use the model for prediction, provide two arrays. The first array should contain user features, which represent the average ratings for tourism places in the 'budaya', 'cagar alam', and 'bahari' categories. The second array should contain tourism features, including price, rating, latitude, longitude, and category encoded as follows:

```python
user_features = [[5.0, 5.0, 1.0], [4.0, 3.0, 4.0], [3.0, 5.0, 0.0]]
tourism_features = [[75000, 4.5, -6.851659, 107.595553, 1], [50000, 4.7, -6.859701, 107.636098, 2], [325, 75000, 4.7, -6.897136, 107.655847, 3]]
```

Before making predictions, please scale the user features and tourism features using the provided scaler. You can load the scaler using the joblib library:

```python
import joblib

scalerTourism = joblib.load('scalerTourism.save')
scalerUser = joblib.load('scalerUser.save')	

scaled_user_features = scalerUser.transform(user_features)
scaled_tourism_features = scalerTourism.transform(tourism_features)
```

Now, you can predict the rating that the user will give to the corresponding tourism places by using the predict method. The prediction will be in a scaled version, so you need to unscale it using the provided scaler:

```python
import joblib
scalerTarget = joblib.load('scalerTarget.save')

prediction = model.predict([scaled_user_features, scaled_tourism_features])
prediction = scalerTarget.reverse_transform(prediction)
```

Finally, you can determine the index of the highest rating to obtain the tourism places recommendation for the user.

## Tools and Library
- Python
- TensorFlow
- NumPy
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn
- Google Collab

## Deployment
We use Flask to deploy a trained Machine Learning model which you can view in [here](https://github.com/C23-PR571-Greenify/Greenify-Predict).

## Reference
