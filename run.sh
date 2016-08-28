date >> results
echo common_neighbors >> results
python link_prediction.py common_neighbors facebook_combined.txt >> results

echo jaccard >> results
python link_prediction.py jaccard facebook_combined.txt >> results

echo adamic_adar >> results
python link_prediction.py adamic_adar facebook_combined.txt >> results

echo preferential_attachment >> results
python link_prediction.py preferential_attachment facebook_combined.txt >> results


#echo katz >> results
#python link_prediction.py katz facebook_combined.txt >> results

#echo hitting_time >> results
#python link_prediction.py hitting_time facebook_combined.txt >> results

#echo rooted_pagerank >> results
#python link_prediction.py rooted_pagerank facebook_combined.txt >> results

#echo friendtns >> results
#python link_prediction.py friendtns facebook_combined.txt >> results
