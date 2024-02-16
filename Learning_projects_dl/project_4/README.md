# drSever Data Science SF
Hello everyone! My name is Aleksandr and I am learning Data Science in Skill Factory.
## Note
I will be grateful for the help and criticism of the projects. And I apologize for my English.

# Project: [YOLOv5 vs. Faster RCNN](../project_3/README.md). 
# Table of contents
1. [Job description](https://github.com/drSever/drSever_data_science/tree/main/Learning_projects_dl/project_4#Job-description)
2. [Quality metric](https://github.com/drSever/drSever_data_science/tree/main/Learning_projects_dl/project_4#Quality-metric)
3. [What we practice](https://github.com/drSever/drSever_data_science/tree/main/Learning_projects_dl/project_4#What-we-practice)
4. [Stages of the project](https://github.com/drSever/drSever_data_science/tree/main/Learning_projects_dl/project_4#Stages-of-the-project)
5. [Result](https://github.com/drSever/drSever_data_science/tree/main/Learning_projects_dl/project_4#Result)

## Job description

Face recognition is a key area in computer vision and pattern recognition. A significant amount of research in the past has contributed to the development of sophisticated face recognition algorithms. With the development of convolutional networks, highly accurate image classification became possible. In the past module, you successfully recognized five IT industry selebrities by training a classifier.
In addition to image partitioning, there are often tasks where a particular object in an image is important. For example, due to the recent COVID19 outbreak, wearing medical masks has become mandatory for everyone outside their homes, which in turn has led to the need to create an approach for detecting masks on people's faces.

Imagine that your company has an opportunity to help the world by training a model to monitor people wearing masks. In doing so, you want to compare the two types of detectors learned (one-level and two-level) and provide a short report to your management team. In order to justify your model choice to your colleagues, you have decided to use the metrics Intersection over Union and Average Precision.

The training data is on the [Kaggle](https://www.kaggle.com/andrewmvd/face-mask-detection]) platform.
The dataset consists of 853 images.

The project is implemented in **Google Colab Pro**.

- *data_yolo.yaml* - .yaml file for YOLO model with dataset information    

## Quality metric

- data loaded using DataLoader
- a pre-trained model is used and its choice is justified in the comments
- RCNN and YOLO family detectors trained
- training takes place on the training sample
- AP on validation sample of two detectors > 0.85
- The results obtained are visualized
- The code is readable and understandable, comments have been added

## What we practice

- solving the problem of detecting objects in the image
- using the principle of transfer of learning
- analyze the performance of algorithms using metrics specific to recognition tasks
- visualization of data and the resulting model predictions

## Stages of the project

- download and upload the dataset
- draw one batch of downloaded data
- choose a pre-trained model and justify your choice in a comment
- train two object detectors using the selected model
- visualize model predictions
- calculate IoU and AP metrics for each model and plot Precision-Recall graphs
- describe your findings in a notebook text box

## Result

Tasks completed, project files uploaded to GitHub. 


