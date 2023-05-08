# Task 1c : Automatic undetermined detection of pygmy blue whale with small number of trainning data

## 1. Introduction 

- Some word about context of such studies cases in ecoacoutsic ...
    - Dificulties in generalization for most of the developped detection models
    - Large dataset
    - Reducing manual annotation 
- Study case where this new task is usefull :
    - I have a new dataset to study a specific specie
    - Already trainned models on other datasets do not works here
    - How can i chose efficiently a small part of my dataset to be manually annotated ?
    - How can i train a network for automatic detection using only a small (but wiselly selected) developpment set ?
- Objectifs :
    - Reduce pre-processing time (Data selection, Manual annotation) while maintaining detection quality
    - Increase unsupervised first analysis of a new dataset
    - Proposed optimized methods to analyse audio underwater dataset 

## 2. Task Descrption  
![Alt text](task_fig2.png?raw=true "Test")


### 2.1 Unsupervised Samples selection : 
- 1500 samples over 25000 given audio semgments.
- This selection is done without informations about the labels.
- The main goal here is to find the most usefull samples to use as developpment set.  

### 2.2 Model Development : 
- Train an automatic detectors using only 1500 selected audio segments as developpment set.
- The developpment set is weakly labelled, with a binary information of absence or presence per sample.
- The model should be able to automatically detect the vocalizations on the valisation set :
    - Input : 50 seconds audio segment 
    - Ouptut : Binary output by segment (presence/absence).

![Alt text](task_fig.png?raw=true "Test")

## 3 Evaluation Metrics : $F_1-$ Score [1]

- The metrics used to evaluate the model is state of the art metrics for automatic detection system.

1) The model is apply on each sample of the evaluation set :
    - True Positif (TP) : the detection system and the manual annotator agree on a presence on the sample
    - False Positif (FP) : The detection system detect a vocalization on the sample but it has been labelised as negative
    - True Negative (TN) : the detection system and the manual annotator agree on an absence on the sample 
    - False Negative (FN) : The detection system do not detect a labelised vocalization on the sample 

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


## 4. More information about the dataset 
Link to the dataset : ....

Dataset informations [2] : 
- Location : Amsterdam and Saint-Paul Island
- Recording period : from February 28th 2018 to April 5th 2018 
- Rcording device : HT192 WB hydrophone mounted on a SeaExplorer underwater glider
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


