<!-- Heading -->
<a id="readme-top"></a>

<br />
<div align="center">
  <h3 align="center">Sensor Based Human Activity Recognition using GNN and RNNs</h3>
</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="">Dataset</a></li>
    <li><a href="">GNN</a></li>
    <li><a href="">RNNs</a></li>
    <li><a href="">References</a></li>
    <!-- <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li> -->
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

The project forecast various indoor and outdoor human activities. The dataset used is KU_HAR dataset released in 2021. This project explores different architectures (RNNs and GNN) to predict human activities using filtered sensor data. In order to understand the effect of window sizes on model's performance, custom datasets of different window sizes were build using raw signal data. As part of feature engineering, Discrete Fourier Transform values i.e the frequency domain values of the sensor data were added to dataset. These features were selected as they can be calculated in real time with least computational resources. As Ku-HAR dataset is unbalanced, the average weighted f1-score was considered to compare the results.

Human Activity Recognition (HAR) is a dynamic domain that investigates methods and techniques for automatically identifying Activities of Daily Living (ADLs). As mobile and wearable devices become increasingly prevalent, HAR finds broad applicability across numerous domains.

Sensor-based HAR and Computer Visionbased HAR are two distinct approaches for recognizing human activities, each with its own set of advantages and limitations. The advantages of using sensor-based HAR are as follows:
* Privacy : This method can be less invasive from a privacy perspective since it does not rely on visual data. This can make it suitable for scenarios where privacy is a concern.
* Low Light and Adverse Conditions: Sensor-based HAR can work effectively in low-light conditions or environments with limited visibility, as it doesn’t rely on visual information.
* Reduced Computational Load: Processing sensor data is generally less computationally intensive compared to computer vision-based approaches.
* Consistency Across Users: Sensor-based HAR is often more consistent across different users since it relies on body movements and sensor data rather than appearance.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Built With

* Tensorflow (for building RNNs)
* Pytorch Geometric was used to build GNN models

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Dataset

Ku-HAR dataset consists of both raw data and sub-samples combined to form a time-domain sub-sample dataset. The data consists of eighteen distinct activity classes.

KU-HAR: An open dataset for heterogeneous human activity recognition [Link](https://www.sciencedirect.com/science/article/abs/pii/S0167865521000933)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## GNN

Graph Neural Networks (GNN) are deep learning models that take input data in the form of graphs. The fundamental idea of GNN is representational learning i.e learning a representation from an input towards a specific task. GNNs has proven to be an effective
method for analysing non-Euclidean data representations.

The GNN model used in this project takes input as a serires of graph and does a graph level prediction. 

### Data Preparation

For HAR classification task, each timestep is represented as Nodes, while edges represent relationships or connections between
these timestep. The graph generated is an homogenous graph as all the nodes are of the same type with same feature set.

Considering a 3 sec segmented dataset, we have each sample containing 300 timesteps. Each timestep corresponds to a vertex in the graph. At each timestep we are recording readings from 3-axis accelerometer and a 3-axis gyroscope. Thus we have 6 features recorded at each timestep. These 6 readings from sensor is the feature set of each vertex. Each vertex is connected to its previous timestep node. Thus we have a line graph where the degree of all the nodes is 2 except for the first and last node in the window.

### Building a Graph Dataset

Once each sample in the dataset has been modeled into a graph, we need to accumulate all the graphs into a graph Dataset. Building graph datasets in PyTorch Geometric (PyG) is necessary for several reasons: 
* Efficient Data Handling: modeling the set of graphs into a graph dataset is an efficient way for handling graph data, making it easier to work with large-scale graphs.
* Graph Dataset structure acts as an iterator thus making it easier to implement and train graph neural networks (GNNs).
* Graph-specific Operations: Graph-specific operations and transformations, such as message passing, graph convolutions, and graph pooling etc can be applied on a graph dataset.

Overall, building graph datasets in PyTorch Geometric provides a streamlined and efficient way to work with graph data, making it easier to develop and deploy graphbased machine learning models.

### Mini Batching for 

For HAR graph classification problem the graphs in the datasets are small with limited number of nodes and edges. Hence to efficiently process the large number of graphs in the dataset, we group the graphs into batches before feeding it to a GNN model to achieve parallelization and to efficiently use the GPU.

