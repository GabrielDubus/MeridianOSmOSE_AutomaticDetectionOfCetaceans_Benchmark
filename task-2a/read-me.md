Automatic detection of NARW vocalizaion using 3 second segments. This will be a binary detector that will classify presense / absense of NARW upcalls.

3 datasets will be made available containing NARW upcalls:

* DCLDE (large)
* GSL 
* EMB 

The developpers will have to train a detector on the DCLDE dataset and test it on the remaining datasets. We want to evaluate models on how well they generalize to unseen data from other regions.

The goal will be to maximize precision / recall and minimize the FPR per hour or recording.

