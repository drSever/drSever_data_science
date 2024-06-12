# drSever Data Science SF
Hello everyone! My name is Aleksandr and I am learning Data Science in Skill Factory.
## Note
I will be grateful for the help and criticism of the projects. And I apologize for my English.

# Project: Style transfer on mobile devices. 
# Table of contents
1. [Job description](https://github.com/drSever/drSever_data_science/tree/main/Learning_projects_dl/project_10#Job-description)
2. [Quality metric](https://github.com/drSever/drSever_data_science/tree/main/Learning_projects_dl/project_10#Quality-metric)
3. [Solution](https://github.com/drSever/drSever_data_science/tree/main/Learning_projects_dl/project_10#Solution)

## Job description

This project asks you to travel to 2016 with 2021-2022 technologies and develop a mobile app similar to Prisma.

I'm a young energetic startup, I'll have to do everything at once while your partner is looking for investment, thinking up a business plan, etc. There is no money to pay for servers, so we decide to run all the model calculations on the device.

**The task** is to train the model, optimise it and import the weights into the app.


## Quality metric

- Availability of plausible model training (to be checked by inferring images from the [repository](https://github.com/magenta/magenta/tree/main/magenta/models/arbitrary_image_stylization#train-a-model-on-a-large-dataset-with-data-augmentation-to-run-on-mobile))
- Availability of an app/web interface/prototype with camera

The app itself is evaluated separately. You can earn additional points if you managed to make a mobile application.

- Video evidence of the prototype working (if the app is not for a phone)
- Link to the app
- Video of the application working on a mobile device

## Solution

The solution is divided into 2 steps:

1. **Obtaining models for inference and use in the mobile application.**
 This solution is implemented on the Google Colab platform.

    - *Project.ipynb* - this notebook presents the implementation of the solution for this stage
    - *images* & *magenta_images* - source images and style images
    - *models* - received models

2. **Creating a mobile application on iOS using the obtained models.**
    - *ios* - implementation of this step     
[Link](https://youtu.be/ypb2P3RHKqA) to the video of the prototype work.