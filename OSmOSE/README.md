# Task B : Automatic undetermined detection of pygmy blue whale with small number of trainning data

## 1. Introduction 

The goal of this task is to propose an efficient workflow for automatic detection of cetaceans vocalization in case of low supervised study, through the example of pygmy blue whale vocalizations recorded in the Indian Ocean.

Based on general observations of the state of the art in the DCLDE community, the perfect detector doesn't exist yet.
In most practical cases, from a new dataset, a manual annotation is still needed in order to constitute a training set and apply a specific trained detector on the rest of the dataset.

The quality of the detection is directly related to 1) the quality of the manual annotation, 2) the number of annotated samples and 3) the representativeness of the training set regarding the whole dataset. 
In number of study cases, the number and the quality of annotated sample is limited as manual annotation is an expensive and time consuming task. Moreover, the annotated part of the dataset is generally randomly selected. 

This task proposed to improve the result of automatic detection in a context of limited, but wisely selected the training set.

- The two main exercices will be to :
    - Select a few subset of the dataset (1500 samples over 25000) through unsupervised method. You are not allowed to use the manual annotation in this part.
    - Once the selection is done, you can use the annotation to train an automatic detection methods for the vocalization of pygmy blue whales. Considering the small number of samples available in the training set, this part is close from to literature of few-shot learning. 


The final objectives of this benchmark will be to establish a state of the art method to optimize the preprocessing time (data selection and manual annotation) while maintaining the quality of the detection in PAM studies applied on cetaceans.


## 2. Task Descrption  
![Alt text](task_fig2.png?raw=true "Test")


### 2.1 Unsupervised Samples selection : 
- Proposed a selection of 1500 50-seconds samples over 25000 through unsupervised methods.
- This selection is done **without** information about the labels.
- The main goal here is to find the most useful samples to use as development set.  

### 2.2 Model Development :
- Once the selection is done, you are allowed to use weak manual annotation.
- The proposed labels are binary for each sample : 0 for *absence* and 1 for *presence* of pygmy blue whale vocalization. 
- Train an automatic detectors using only 1500 selected audio segments as development set.
- The model should be able to automatically detect the vocalizations on the validation set :
    - Input : 50 seconds audio segment 
    - Output : Binary output by segment (presence/absence).

![Alt text](task_fig.png?raw=true "Test")

## 3 Evaluation Metrics : $F_1-$ Score [1]

- The metrics used to evaluate the model are state of the art metrics for automatic detection systems.

1) The model is apply on each sample of the evaluation set :
    - True Positif (TP) : the detection system and the manual annotator agree on a presence on the sample
    - False Positif (FP) : The detection system detect a vocalization on the sample but it has been labeled as negative
    - True Negative (TN) : the detection system and the manual annotator agree on an absence on the sample 
    - False Negative (FN) : The detection system do not detect a labeled vocalization on the sample 

2) $F_1$ score is computed using $precision$ and $recall$ :

$\begin{equation}
    precision = \dfrac{TP}{TP+FP}, ~~~~ recall = \dfrac{TP}{TP+FN}
\end{equation}
$

and, 

$\begin{equation}
    F_1 = 2\dfrac{precision \cdot recall}{precision + recall}
\end{equation}
$

The evaluation is made using 25000 samples. Those samples are not available in the unsupervised samples selection. 

## 4. More information about the dataset 
Link to the dataset : ....

Dataset informations [2] : 
- Location : Amsterdam and Saint-Paul Island
- Recording period : from February 28th 2018 to April 5th 2018 
- Recording device : HT192 WB hydrophone mounted on a SeaExplorer underwater glider
- Sampling Rate : 240Hz
- Sound event annotated : Pygmy Blue Whale Vocalization


## 4. Bechmark methods :

### Samples selection : Random selection 
### Model Development : 
- 2 Models used :
    - ResNet18 [3]
    - Small 3CNN+3FC network 

![Alt text](Network_Schema.png?raw=true "Test")


### Apply benchmark model on validation set :

- You needs to change the files path_codes.txt with the absolute path of the codes
- You needs to change the files path_osmose_dataset.txt with the absolute path of the location of the dataset folder

- Results :


## 4. References :

[1] : https://en.wikipedia.org/wiki/F-score

[2] : Torterotot, Maëlle, Julie Béesau, Cécile Perrier de la Bathie, et Flore Samaran. « Assessing Marine Mammal Diversity in Remote Indian Ocean Regions, Using an Acoustic Glider ». Deep Sea Research Part II: Topical Studies in Oceanography 206 (décembre 2022): 105204. https://doi.org/10.1016/j.dsr2.2022.105204.

[3] : He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).


