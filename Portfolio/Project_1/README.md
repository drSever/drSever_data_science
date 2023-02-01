# drSever Data Science SF
Hello everyone! My name is Aleksandr and I am learning Data Science in Skill Factory.
## Note
I will be grateful for the help and criticism of the projects. And I apologize for my English.

# Project: Classification task. 
# Table of contents
1. [Job description](https://github.com/drSever/drSever_data_science/tree/main/Portfolio/Project_1#Job-description)
2. [Quality metric](https://github.com/drSever/drSever_data_science/tree/main/Portfolio/Project_1#Quality-metric)
3. [Stages of the project](https://github.com/drSever/drSever_data_science/tree/main/Portfolio/Project_1#Stages-of-the-project)
4. [Result](https://github.com/drSever/drSever_data_science/tree/main/Portfolio/Project_1#Result)

## Job description

Banks would like to be able to choose among their customers those who are most likely to take advantage of this or that offer, and to contact them.
We are presented with data from the last marketing campaign that the bank conducted: the task was to attract clients to open a deposit. We need to analyze this data, identify the pattern and find the decisive factors that influenced the client to invest in this particular bank. If we can do this we will raise the bank's income and help us understand the target audience which we need to attract by means of advertising and various offers.

**Business task**: to define the characteristics which can be used to identify the clients who are more inclined to open a deposit in the bank and due to this to increase the effectiveness of marketing campaign.

**Technical task**: to build a machine learning model that will predict, based on the proposed characteristics of the client, whether he will use the offer to open a deposit or not.

Dataset can be downloaded [here](https://lms.skillfactory.ru/assets/courseware/v1/dab91dc74eb3cb684755123d224d262b/asset-v1:SkillFactory+DST-3.0+28FEB2021+type@asset+block/bank_fin.zip)

## Quality metric

Built a machine learning model that will predict, based on the proposed characteristics of the client, whether or not he will take advantage of the offer to open a deposit.

## Stages of the project

- **Iteration #1** - creating Baseline, defining a strategy for dealing with gaps in the data. Only simple and necessary data processing was performed. No setup of hyperparameters of models was performed.
- **Iteration #2** - full-fledged EDA was performed, new features were created.
- **Iteration #3** - processing of outliers, selection of features, tuning of hyperparameters of models using Optuna with cross validation.


## Result

- Models of logistic regression, random forest, gradient boosting and stacking were trained.
- Improved model metrics compared to Baseline. 
- the following resources were used for model training: MacBook m1 pro, Google Colab (free), Yandex DataSphere (trial), SaturnCloud (free).
- project files uploaded to GitHub. 


