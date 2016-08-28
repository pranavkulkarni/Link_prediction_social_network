Project: Link Prediction In Social Networks - People you may know!
Team no. 24
Goal of the project: Given a snapshot of a social network, we want to infer which new interactions among its members are likely to occur in the near future based on network topology. Our goal is to understand which measures of “proximity” in a network lead to the most accurate link predictions.

The program to execute is link_prediction.py and the dataset is Facebook data which is an edge list which is present in the data folder.

Dependencies: igraph package
Installing igraph for Python
Assuming you are using Ubuntu and Anaconda distribution,
if pip is not installed:
conda install pip
then:
pip install python-igraph

******************************************************************************************************************************************************
Running instructions:
python link_prediction.py <common_neighbors/jaccard/adamic_adar/preferential_attachment/katz/friendtns> data_file_path

Examples:
python link_prediction.py common_neighbors data/facebook_combined.txt
python link_prediction.py jaccard data/facebook_combined.txt
python link_prediction.py adamic_adar data/facebook_combined.txt
python link_prediction.py preferential_attachment data/facebook_combined.txt
python link_prediction.py katz data/facebook_combined.txt
python link_prediction.py friendtns data/facebook_combined.txt

(note: katz algorithm takes about 1 hr to run as it is a path based global method. Other methods take about 3-5 mins.)
******************************************************************************************************************************************************

Brief explanation of the program:
We have implemented 6 methods for link prediction in social network.
Common Neighbors, Jaccard’s Coefficient, Adamic/Adar, and Preferential Attachment are local based similiarity(proximity) measures.
Katz is a global based (path based) similiarity measure.
FriendTNS is a combination of local and global based similarity measures.
Details about each method is explained in the ppt/slides.

The basic approach of the program is as follows:
1. Read the data set which is an edge list (graph).
2. Divide the edge list into train and test data (50 % each)
3. Build an igraph on the training data and apply each algorithm based on the argument based to the script.
4. A similarity matrix is calculated for every pair of nodes x,y in the graph.
5. Use this similarity matrix to recommend top k nodes as "people you may know" for a target user present in the test set.