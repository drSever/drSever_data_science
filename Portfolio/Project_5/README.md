# drSever Data Science SF
Hello everyone! My name is Aleksandr and I am learning Data Science in Skill Factory.
## Note
I will be grateful for the help and criticism of the projects. And I apologize for my English.

# Project: Civil Society Research Lab Brief. Identification of vulnerable groups. 
# Table of contents
1. [File description](https://github.com/drSever/drSever_data_science/tree/main/Portfolio/Project_5#File-description)
2. [Job description](https://github.com/drSever/drSever_data_science/tree/main/Portfolio/Project_5#Job-description)
3. [Quality metric](https://github.com/drSever/drSever_data_science/tree/main/Portfolio/Project_5#Quality-metric)
4. [Stages of the project](https://github.com/drSever/drSever_data_science/tree/main/Portfolio/Project_5#Stages-of-the-project)
5. [Result](https://github.com/drSever/drSever_data_science/tree/main/Portfolio/Project_5#Result)

## File description

- *1_Data_collection_and_preparation.ipynb* - first stage of work: collection, processing and preliminary analysis of the submitted data, creation of a summary table
- *2_FE_EDA_FS.ipynb* - the second stage of the work is the creation of new features, EDA, selection of features for clustering
- *3_Clustering.ipynb* - the third stage of work: clustering of regions, description of clusters, answers to the questions posed
- *project_module.py* - separate module with necessary functions and standardized region names
- *conda_env.yml* - if you use Anaconda (Conda)
- *requirements.txt* - if you use 'pip'
#
- *./project_chatbot* - solution realization in production (chat bot)       
- *./data_provided* - presented data to solve the clustering task      
- *./data_output* - output data obtained during the execution of the task:       
     - *data_final* - final summary table with features       
     - *data_final_clust* - final summary table with cluster labels       
     - *features_for_clustering.json* - features selected for clustering        
     - *.svg* and *.html* - polar diagram files for presentation in GitHub     

## Job description

We have data on income, morbidity, socially vulnerable segments of the Russian population and other economic and demographic data at our disposal.

**Our task:**
- cluster Russia's regions and determine which of them are most in need of assistance to low-income/disadvantaged segments of the population;
- describe population groups facing poverty;
- determine:
    - whether the number of children, pensioners and other socially vulnerable groups affects the level of poverty in the region;
    - whether the level of poverty/social disadvantage is related to production and consumption in the region;
    - what other dependencies can be observed in relation to socially disadvantaged segments of the population.

## Quality metric

To perform clustering of Russian regions, which will be effective in identifying regions that are in dire need of assistance to low-income/disadvantaged groups of the population. Answer the questions posed.

## Stages of the project

- Collection and processing of submitted data
- Downloading and processing of additional data
- Data processing and preliminary analysis
- feature creation (Feature Engineering)
- EDA
- Feature Selection
- Clustering
- Description of the clusters and answers to the questions posed

## Result

- Clustering performed
- Answers to the questions posed
- The solution is realized in production (chek the folder *./project_chatbot*)
- Project files uploaded to GitHub. 


