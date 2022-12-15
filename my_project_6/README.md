# drSever Data Science SF
Hello everyone! My name is Aleksandr and I am learning Data Science in Skill Factory.
## Note
I will be grateful for the help and criticism of the projects. And I apologize for my English.

# My project 6. ML-7. Optimization of model hyperparameters.
# Table of contents
1. [Job description](https://github.com/drSever/drSever_data_science/tree/main/my_project_6#Job-description)
2. [Quality metric](https://github.com/drSever/drSever_data_science/tree/main/my_project_6#Quality-metric)
3. [What we practice](https://github.com/drSever/drSever_data_science/tree/main/my_project_6#What-we-practice)
4. [Stages of the project](https://github.com/drSever/drSever_data_science/tree/main/my_project_6#Stages-of-the-project)
5. [Result](https://github.com/drSever/drSever_data_science/tree/main/my_project_6#Result)

## Job description

Our practice will be based on the competition [Kaggle: Predicting a Biological Response](https://www.kaggle.com/c/bioresponse). 
The [data](https://lms.skillfactory.ru/assets/courseware/v1/9f2add5bca59f8c4df927432d605fff3/asset-v1:SkillFactory+DST-3.0+28FEB2021+type@asset+block/_train_sem09__1_.zip) can be downloaded here. 

It is necessary to predict the biological response of molecules (column 'Activity') by their chemical composition (columns D1-D1776).
The data are presented in CSV format.  Each line represents a molecule. 
- The first Activity column contains experimental data describing the actual biological response [0, 1]; 
- The remaining columns D1-D1776 represent molecular descriptors, which are computable properties that can capture some characteristics of a molecule, such as size, shape, or element composition.
No preprocessing is required, the data is already coded and normalized. We will use F1-score as the metric.   

Two models need to be trained: a logistic regression and a random forest. Next, we need to make a selection of hyperparameters using basic and advanced optimization methods. It is important to use all four methods (GridSeachCV, RandomizedSearchCV, Hyperopt, Optuna) at least once each, the maximum number of iterations should not exceed 50.  

## Quality metric

- two models were trained; 
- hyperparameters were selected using four methods; 
- cross validation was used

## What we practice

- classification problem solving
- logistic regression and random forest methods
- cross-validation method
- optimization of hyperparameters
- work with GridSearchCV, RandomizedSearchCV, Hyperopt, Optuna

## Stages of the project

- introduction to the dataset
- check for class balance
- training of logical regression and random forest models on default parameters
- hyperparameter selection
- summarizing and drawing conclusions

## Result

- Task completed, two models were trained; hyperparameters were selected using four methods; cross validation was used, notebook uploaded to GitHub.
- **Mentor hasn't checked my work yet.**


