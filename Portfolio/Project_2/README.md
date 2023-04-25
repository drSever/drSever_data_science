# drSever Data Science SF
Hello everyone! My name is Aleksandr and I am learning Data Science in Skill Factory.
## Note
I will be grateful for the help and criticism of the projects. And I apologize for my English.

# Project: Regression task. 
# Table of contents
1. [Job description](https://github.com/drSever/drSever_data_science/tree/main/Portfolio/Project_2#Job-description)
2. [Quality metric](https://github.com/drSever/drSever_data_science/tree/main/Portfolio/Project_2#Quality-metric)
3. [Stages of the project](https://github.com/drSever/drSever_data_science/tree/main/Portfolio/Project_2#Stages-of-the-project)
4. [Result](https://github.com/drSever/drSever_data_science/tree/main/Portfolio/Project_2#Result)

## Job description

We have a machine learning problem to solve, aimed at automating business processes. We will build a model that will predict the total duration of a cab ride in New York City.   

It is known that the cost of a cab in the U.S. is based on a fixed rate and a fare, the amount of which depends on time and distance. Fares vary from city to city.    
In turn, the time of the trip depends on many factors, such as the direction of the trip, time of day, weather conditions and so on.    
Thus, if we develop an algorithm capable of determining the duration of a trip, we can predict its cost in the most trivial way, for example, simply by multiplying the cost by a given fare.    

The task we will be solving was presented as a Data Science [competition](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/overview) with a prize of $30,000 on the Kaggle platform in 2017.    
We will be given a data set containing information about yellow cab rides in New York City for 2016. The data was originally released by the New York City Taxi and Limousine Commission and includes trip times, geographic coordinates, number of passengers, and several other variables.

**Business task:** determine the characteristics and use them to predict the length of a cab ride.    
**Technical task:** to build a machine learning model that will predict the numerical attribute - the cab ride time - based on the proposed characteristics of the client, that is, to solve the regression problem.    

Dataset can be downloaded [here](https://drive.google.com/file/d/1X_EJEfERiXki0SKtbnCL9JDv49Go14lF/view?usp=sharing).    

## Quality metric

Built a machine learning model that will predict the numerical attribute - the cab ride time - based on the proposed characteristics of the client, that is, to solve the regression problem.

## Stages of the project

- Generate a data set based on several sources of information.
- Design new features using Feature Engineering and identify the most relevant ones when building the model.
- Examine the data provided and identify patterns.
- Build several models and choose the one that shows the best result for a given metric.
- Design a process for predicting the duration of a trip for new data.
- Upload your solution to the Kaggle platform, thereby participating in a real Data Science competition.


## Result

- Initial data on cab trips were examined and cleaned up.
- Additional data were added (OSRM, weather, etc.)
- Performed by EDA and FE
- Were found and marked outliers in the data. 
- The selection of features was made.
- LinearRegression, DecisionTreeRegressor (including hyperparameter selection), RandomForestRegressor, GradientBoostingRegressor, XGBoost, CatBoost, Stacking model are trained.
- The selection of hyperparameters was carried out only on the decision tree model, because on the other models it was too resource-intensive.
- It was noted that the importance of the features.
- The following results were obtained in the Kaggle competition:
    - Public Score 0.42
    - Private Score 0.41
- Project files uploaded to GitHub. 


