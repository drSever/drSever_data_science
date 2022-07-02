# drSever Data Science SF
Hello everyone! My name is Aleksandr and I am learning Data Science in Skill Factory.
## Note
I will be grateful for the help and criticism of the projects. And I apologize for my English.

# My project 3. Section PYTHON-14. Data cleaning. Consolidation of knowledge. 
# Table of contents
1. [Job description](https://github.com/drSever/drSever_data_science/tree/main/my_project_3#Job-description)
2. [Quality metric](https://github.com/drSever/drSever_data_science/tree/main/my_project_3#Quality-metric)
3. [What we practice](https://github.com/drSever/drSever_data_science/tree/main/my_project_3#What-we-practice)
4. [Stages of the project](https://github.com/drSever/drSever_data_science/tree/main/my_project_3#Stages-of-the-project)
5. [Result](https://github.com/drSever/drSever_data_science/tree/main/my_project_3#Result)

## Job description

There are [two databases](https://lms.skillfactory.ru/assets/courseware/v1/958d35ff25f2486f65613da4459e6647/asset-v1:SkillFactory+DST-3.0+28FEB2021+type@asset+block/Data_TSUM.xlsx) (two sheets of Excel file): a database with competitor prices (Data_Parsing) and an internal company database (Data_Company).  
There are two id's in the parsing database that uniquely identify the product: ***producer_id*** and ***producer_color***.  
There are two similar fields in the company database: ***item_id*** and ***color_id***.  
We know that the codes in the two databases differ in the presence of a set of service characters. The following characters can be found in the parsing base: _, -, ~, \\, /.  

## Quality metric

- Read data from Excel in DataFrame (**Data_Parsing**) and (**Data_Company**).
- Pull the data from the company database (***item_id, color_id, current_price***) to the parsing database and form a column of price difference in % (competitor's price to our price).
- Identify strong deviations from the average in the price difference within a brand-category (that is, remove random outliers that strongly distort the comparison). Criterion - by taste, write a comment in the code.
- Write a new Excel file with the parsing base, with columns from step 2 glued to it and taking into account step 3 (you can add an ***outlier*** column and put **Yes** for outliers).

## What we practice

- Read data from Excel files.
- The ability to work with data and transform it.
- Remove special characters from feature values in DataFrame. 
- Ability to work with visualization libraries. 
- Ability to work with documentation. 
- Save data to Excel files.

## Stages of the project

- I downloaded the data from an Excel file.
- I did data cleaning, data merging.
- I did an exploratory analysis of the data (EDA). 
- I made diagrams. 
- I found strong deviations from the average in price differences within a category brand and noted them.
- I saved the data into a new Excel file.
- My mentor checked my work. I corrected the mistakes.

## Result

Tasks completed, project files uploaded to GitHub.  

