# MIDAS
**MI**tochondrial **D**isease **A**ssociation **S**oftware - *Using machine learning on interaction network topology to diagnose genetic diseases in patient genomes.*

# Abstract
The genetic diagnosis of mitochondrial disease patients by exome sequencing has a low success rate of ~30%. We considered whether it would be possible to make use of interaction network data  describing functional and physical interactions between proteins to identify from a candidate gene proximate to known mitochondrial disease genes and therefore highlight it as potentially being the pathogenic gene in that patient. 

We present a network topology & machine learning based method that decomposes interaction networks into communities using network topology and exposes the cluster membership profile of these genes to a machine learning classifier while also determining the optimal number of communities into which the network should be partitioned. Out method shows good machine learning performance of 0.91 ROC AUC during 10 fold cross validation when tested against known mitochondrial disease genes which is reproducible across multiple classifier types and network community numbers. Preliminary results also show good agreement with the predictions of human scientists for the disease causing gene in a mitochondrial disease patient.

Our method may also be generalizable to a range of disease types by changing the training data used to train the machine learning classifier and we are in the process of investigating this line of research.

# Steps in the process:
## 1. Set the interaction network to use
Currently only supports STRING but user can filter the graph based on edge scores / type before starting MIDAS process
## 2. Set the training data
The training data is a list of IDs, the ids must be ENSP ids to match the STRING network ids, the "ENSP" must be removed from the begining of ids to make them numerical to allow for indexing nodes in the network as integers (a limitation of BIGCLAM community detection algorithm)
## 3. Discover the best number of network communities to use
Machine learning performance is affected by partitioning the network into a different number of communities although this variation is minimal beyond a small number of communities suggesting that machine learning performance is not an artifact of overfitting.
## 4. Evaluate the results
The pipeline will create folders of the form `string_N_communities` where `N` is the number of communities tested for the machine learning process. These folders contain the plots and machine learning cross validation scores for the user to evaluate and each folder contains the machine learning score predictions for all test data genes in the file `class_prediction_scores_rand_forest.txt` (assuming the default random forest classifier was used). The evaluation of results will be improved in the coming refactor.

# Usage:
Currently due to some glue code in bash for the example mitochondrial disease case simply run
`$ ./run_all_cluster_numbers.sh`

To change training data or network these variables must be changed manually in `run_for_n_clusters.sh` and `ml_pipeline_graph_clusters.py` respectively.

A refactor is in process which will make the running process more userfriendly.