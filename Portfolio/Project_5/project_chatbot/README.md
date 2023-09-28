# Project chat bot

The bot allows you to retrieve the following information:
- enter the name of the region and find out in which cluster it is located
- get a general description of the cluster
- enter the name of the attribute, get general information on the attribute in the region, as well as get a diagram of values of this attribute in this cluster by regions

## File description

- *./src* - chatbot folder:
    - *output* - folder with diagrams of the feature by cluster regions
    - *chatbot.py* - the chat bot itself
    - *data_final_cluster* - table with summary attributes and cluster labels

## Commands

- *docker build -t chatbot_image .* - to build the image
- *docker run -it --rm chatbot_image* - to run the image
#
- *docker pull drsever/chatbot_image* - to download a ready-made image