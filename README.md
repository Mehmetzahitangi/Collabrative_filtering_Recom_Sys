Subject: User-based and item-based collabrative filtering recommendation system


Purpose of the project:
The rapid growth of data collection has led to a new era of information. Data is being used to create more efficient systems and this is where Recommendation Systems come into play. Recommendation Systems are a type of information filtering systems as they improve the quality of search results and provides items that are more relevant to the search item or are realted to the search history of the user. They are used to predict the rating or preference that a user would give to an item. In this project, our purpose is creating a recommendation system that predict all movies in test dataset based on user-based or item-based collabrative filtering.


Collabrative Filtering: The recommendations get filtered based on the collaboration between similar user’s or item’s of  preferences. If there is collaboration between similar user’s , that is user-based collabrative filtering. If there is collaboration between similar item’s , that is item-based collabrative filtering. 
In codes, there is the process that is user-based or item-based according to user that use the this python program. If the user selects item-based, the program will calculate cosine similarity. This calculation process imported sklearn library because improved libraries ,like sklearn , works/calculates better. Thus, results will be more correct. If the user selects user-based, the program will calculate pearson correlation. Due to the same reason, the calculation pearson correlation  process imported sklearn library,too. 



DATA OVERVİEW
After remove timestamp column in dataset:

Number of users = 943 | Number of movies = 1682

           	 user_id        		item_id              rating
count    100000.00000     100000.000000  	   100000.000000
mean    462.48475            425.530130      	  3.529860
std         266.61442    	     330.798356      	  1.125674
min        1.00000     	     1.000000       	  1.000000
25%       254.00000  	     175.000000     	   3.000000
50%       447.00000	     322.000000     	   4.000000
75%       682.00000 	     631.000000      	  4.000000
max       943.00000	     1682.000000    	    5.000000


First 10 data
     user_id      item_id    rating
0      196      	242        	3
1      186      	302        	3
2       22      	377        	1
3      244       	51        	2
4      166      	346        	1
5      298      	474        	4
6      115      	265        	2
7      253      	465        	5
8      305     	 451        	3
9        6       	86        	3

Experimental methodology
k-fold cross validation: It is one of the methods of separating the data set into parts for evaluating the classification models and traning the model. 
On the program, the user determines how many parts the data set will be split, from four to ten and all proccesses will be execute for each separated step. 
The the evaluation metric results that calculated are print for each separated step, too.


Experimental results 
The Evaluation Metrics
MAE – Mean Absolute Error and MSE – Mean Squared Error 

User-based and k-folds is 5:
user-based mean absolute error:  0.26354618616064834
user-based mean squared error: 0.24715690635726056

user-based mean absolute error:  0.2616453043336475
user-based mean squared error: 0.2413300039544679

user-based mean absolute error:  0.2622214486773795
user-based mean squared error: 0.24009493408960378

user-based mean absolute error:  0.2624899715353472
user-based mean squared error: 0.24091205709521285

user-based mean absolute error:  0.26393237117550233
user-based mean squared error: 0.2452751756722422
General Mean MAE for all iterations:  0.262767056376505
General Mean MSE for all iterations:  0.24295381543375746

Item-based and k-folds is 3:
item-based mean absolute error:  0.20001769084087828
item-based mean squared error: 0.2917242378634789
item-based mean absolute error:  0.20055983528084426
item-based mean squared error: 0.2888718814106754
item-based mean absolute error:  0.19904607496183885
item-based mean squared error: 0.28590201282253763
General Mean MAE for all iterations:  0.19987453369452046
General Mean MSE for all iterations:  0.28883271069889727

Visualization

https://github.com/Mehmetzahitangi/Collabrative_filtering_Recom_Sys/blob/master/Figure%202021-01-04%20174747.png

