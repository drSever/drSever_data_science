# drSever Data Science SF
Hello everyone! My name is Aleksandr and I am learning Data Science in Skill Factory.
## Note
I will be grateful for the help and criticism of the projects. And I apologize for my English.

# Project: Clusterisation task. 
## Clustering of vehicle images. Real case from IntelliVision

## Files description

- **project_IntelliVision.ipynb** - file contains a clustering of all descriptors (**not displayed on GitHub**)
- **project_IntelliVision_1.ipynb** - file contains a clustering of the following descriptors: *efficientnet-b7*, *osnet* 
- **project_IntelliVision_2.ipynb** - file contains a clustering of the following descriptors: *vdc_color*, *vdc_type*

## Task description.
There is a set of 416,314 images of vehicles of different types, colors and taken from different angles.

The IntelliVision team has already processed its data set using several deep learning models (convolutional neural networks) and obtained four feature vectors (descriptors) for each image.

**Our task:** using the ready descriptors, we split the images into clusters and interpret each of them. For all variants of descriptors we need to apply several clustering algorithms and compare the results. You can compare based on metrics, visualizations of cluster densities, and how well the clusters are interpreted.

**An additional subtask** is to find outliers among the images. These could be images of poor quality, images with glare, or images with no vehicles, etc.

**Business task:** to investigate the possibility of using clustering algorithms for partitioning new data and finding outliers.

**Technical task:** to build a model for image clustering based on descriptors extracted by different neural network architectures, to interpret the results and to choose a model or a combination of models that extracts the most suitable features for interpretation.

## Main objectives:
For each type of descriptor, you must:
- perform preprocessing of descriptors;
- cluster images based on their descriptors, selecting an algorithm and clustering parameters;
- visualize the obtained clusters in 2D or 3D space;
- Interpret the obtained clusters - in a couple of sentences formulate what images are in each of the clusters.
- Compare the obtained clustering for each type of descriptors (by metrics, visualization and interpretation results).
- Perform automated search for outliers among images based on descriptors.
- Additional task (not evaluated): try to use a mixture of descriptors obtained by different models and interpret the results.
The artifacts of the work should be a Jupyter Notebook with a detailed study, as well as a CSV file with the best clustering and outlier separation results.

## Intermediate results.
- The work was performed on the Google Colab Pro platform.
- Four sets of descriptors were used separately.
- The following dimensionality reduction algorithms were used: PCA, SVD.
- The following clustering algorithms were used: KMeans, Agglomerative Clustering.
- To evaluate the results of clustering algorithms the following internal measures were used: Calinski-Harabasz Index, Davies-Bouldin Index.
- AgglomerativeClustering shows poor clustering results in all experiments.
- DBSCAN algorithm was used to find outliers.
- Clustering based on **vdc_color** and **vdc_type** descriptors is most effective.
- An interesting observation: when trying to search for outliers in vdc_type descriptors, the DBSCAN algorithm clusters vehicles by color and make.

### Download updated [vdc_color](https://drive.google.com/file/d/1-QiUyRXGk_JXaToIgJF-XNIu7QG-8p1N/view?usp=sharing) and [vdc_type](https://drive.google.com/file/d/1-YF_s9X6ZtC_Tsj8CpzTXUZnucyhnWvl/view?usp=sharing) descriptor data.    
Final **vdc_color** dataset, **cluster** feature:     
- 0 cluster: light-colored cars (mostly white and light gray)
- 1 cluster: other types of dark-colored vehicles.
    
Final dataset **vdc_color**, **dbscan** feature:    
- clusters -1,0,1: basic data
- clusters 2-12: outliers.
    
Final dataset **vdc_type**, **cluster** feature:    
- 0 cluster: large vehicles (station wagons, minivans, parkettes and SUVs, vans, etc.)
- 1 cluster: regular cars.
      
Final dataset **vdc_type**, **dbscan** feature:    
- clusters -1: basic data
- clusters 0-25: outliers

## Conclusions:
- Considering that vdc_color and vdc_type descriptors showed the best clustering results, I believe it is reasonable to use their combination for further experiments.
- It is necessary to take a more in-depth approach to the selection of DBSCAN algorithm parameters for searching for outliers.
- Perhaps it makes sense to try different combinations of clustering algorithms, for example, when searching for outliers in vdc_type descriptors, the DBSCAN algorithm shows good results in grouping similar images with vehicles of the same color and brand.
