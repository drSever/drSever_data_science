# drSever Data Science SF
Hello everyone! My name is Aleksandr and I am learning Data Science in Skill Factory.
## Note
I will be grateful for the help and criticism of the projects. And I apologize for my English.

# Project: Time series.
# Table of contents
1. [Job description](https://github.com/drSever/drSever_data_science/tree/main/Learning_projects/project_9#Job-description)
2. [Quality metric](https://github.com/drSever/drSever_data_science/tree/main/Learning_projects/project_9#Quality-metric)
3. [What we practice](https://github.com/drSever/drSever_data_science/tree/main/Learning_projects/project_9#What-we-practice)
4. [Stages of the project](https://github.com/drSever/drSever_data_science/tree/main/Learning_projects/project_9#Stages-of-the-project)
5. [Result](https://github.com/drSever/drSever_data_science/tree/main/Learning_projects/project_9#Result)

## Job description

Imagine that you work for a government company that provides economic analysis of the world on behalf of the government.

You are tasked with analyzing the GDP of the African country Ghana. To do this, you have been given [Ghana's GDP](https://lms-cdn.skillfactory.ru/assets/courseware/v1/cf3fb9ca311981f5cc6b6f0a40621388/asset-v1:SkillFactory+DST-3.0+28FEB2021+type@asset+block/ghana_gdp.zip) figures for 62 years. You have to investigate the time series, study its properties, build models and draw conclusions from the results. 

## Quality metric

- The time series was analyzed for trend and seasonality and checked for stationarity.
- The choice of the model is justified.
- Interpolation performed.
- The volatility was calculated, the ARCH/GARCH model was applied.
- The results are compared with the results of linear regression.
- The results are correctly validated.


## What we practice

Work with statistical models. Interpolation and validation of time series.  

## Stages of the project

- Read the original data file. Visualize the original time series, make initial conclusions about the presence of trend components and seasonality in the series. Set aside the last three years from the dataset as a test sample to evaluate the prediction results.
- Build a moving average model, analyze the result.
- Use the Dickey-Fuller test to evaluate the stationarity of the time series and decide on the ARMA/ARIMA model.
- Construct an ARMA/ARIMA model to predict the behavior of the time series. Also build several models with parameters closest to the found p and q, and compare the AIC coefficient. 
- Build a model with the best parameters and make predictions for the deferred test sample (last three years).
- Display the result graphically - plot the true and predicted behavior of the time series, as well as the 95% confidence interval for the prediction.
- Draw conclusions from the results.
- Check the data for missing dates.
- Check the data for missing values.
- Perform interpolation using the .interpolate() method to fill in the gaps.
- Check the resulting series for stationarity, define the model parameters (ARIMA/ARMA) and run the model.
- Draw conclusions from the results.
- Calculate the volatility for your time series.
- Use the GARCH model to predict volatility.
- Use linear regression to get a prediction.
- Visualize the resulting prediction along with the actual value.
- Compare the results and draw conclusions.

## Result

- Task completed, notebook uploaded to GitHub.


