# Task 1c : Automatic undetermined detection of pygmy blue whale with small number of trainning data

Nb : You needs to change the files path_codes.txt with the absolute path of the codes
Nb2 : You needs to change the files path_osmose_dataset.txt with the absolute path of the location of the dataset folder


## Task Description : 
Input : 50 seconds audio segment as .wav 

Ouptut : Binary output by segment (presence/absence).

![Alt text](task_fig.png?raw=true "Test")

Evaluation Metrics : $F_1-$ Score [1]

$F_1 = 2\dfrac{precision \cdot recall}{precision + recall}$

with : $precision = \dfrac{TP}{TP+FP}$ and $recall = \dfrac{TP}{TP+FN}$ 

Dataset : 
- 50000 50-seconds samples
    - 25000 samples available for trainning, but you need to select only 1500 of them.
    - 25000 others examples for evaluation.



## Information about the dataset 
Link to the dataset : ....

Dataset informations [2] : 
- Location : Amsterdam and Saint-Paul Island
- Recording period : from February 28th 2018 to April 5th 2018 
- Rcording device : HT192 WB hydrophone mounted on a SeaExplorer underwater glider
- Sampling Rate : 240Hz
- Sound event annotated : Pygmy Blue Whale Vocalization


## Bechmark methods :
Random selection 

- 2 Models used :
    - ResNet18 [3]
    - Small 3CNN+3FC network :

![Alt text](Network_Schema.png?raw=true "Test")


- Results :



[1] : https://en.wikipedia.org/wiki/F-score

[2] : Torterotot, Maëlle, Julie Béesau, Cécile Perrier de la Bathie, et Flore Samaran. « Assessing Marine Mammal Diversity in Remote Indian Ocean Regions, Using an Acoustic Glider ». Deep Sea Research Part II: Topical Studies in Oceanography 206 (décembre 2022): 105204. https://doi.org/10.1016/j.dsr2.2022.105204.

[3] : He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).


