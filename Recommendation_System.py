# All libraries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt


headers = ["user_id","item_id","rating","timestamp"]

file_type = input("Which type is your data? 1-xlsx or 2-csv. Please select 1 or 2")

if file_type == "1":
    
    # Read and store content 
    # of an excel file  
    path_ = input("Please write absolute path your data ") 
    
    read_file = pd.read_excel(path_) 
    print("Data converted and saved as csv file in your directory")
    
    # Write the dataframe object 
    # into csv file 
    path = input("Please write absolute path your csv file that created ") 
    
    read_file.to_csv(path) 
    data = pd.read_csv(path,names=headers) 
    
elif file_type == "2":

    path = input("Please write absolute path your data ") 
    
    data = pd.read_csv(path,names=headers) 
    
else:
    print("exiting... Please rerun and write 1 or 2. 1 == xlsx and 2 == csv")
    
# read csv file and convert  
# into a dataframe object 
#data = pd.read_csv("data.csv",names=headers) 
data.reset_index(drop=True, inplace=True)
data = data.drop(["timestamp"],axis=1)

n_users = data.user_id.unique().shape[0]
n_items = data.item_id.unique().shape[0]

print("Number of users = " + str(n_users) + " | Number of movies = " + str(n_items))
print(data.head())
print(data.describe())

data_numpy = data.to_numpy()




fold_number = int(input("Please write a fold number from 4 to 10 "))
while True:
    if fold_number <4 or fold_number > 10:
        print("The number is not between 4 and 10, please try again")
        fold_number = int(input())
    else:
        break

kf = KFold(n_splits=fold_number)

model = input("Write a model that you want to use, two options: item or user ")
while True:
    if model != "item" and model != "user":
        print("Please write item or user")
        model = input()
    else:
        break
        
        
def visualize(absolute_error,squared_error):
    
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle('Evaluate Metrics - K Fold')
    
    ax1.plot(range(1,fold_number+1), absolute_error, 'o-')
    ax1.set_ylabel('Mean Absolute Error')
    ax1.grid()
    
    ax2.plot(range(1,fold_number+1), squared_error, '.-')
    ax2.set_xlabel('Fold Number')
    ax2.set_ylabel('Mean Squared Error')
    ax2.grid()
    plt.show()

    
def predict(ratings,similarity,type="user"):
    if type == "user":

        mean_user_ratings = ratings.mean(axis=1).reshape(-1,1)
        ratings_diffs = (ratings - mean_user_ratings)

        
        pred = mean_user_ratings + similarity.dot(ratings_diffs) / np.array([np.abs(similarity).sum(axis=1)]).T
        
    elif type == "item":
        pred = (ratings.T.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])).T 
        # similarity array dimension is 1682*943 and  will execute error in print part because dimension is 943*1682 in print part .For this reason we take transpose of this array.
        
    return pred


mean_of_mae = 0 
mean_of_mse = 0
error_list = []
error_list_two = []
for train_index, test_index in kf.split(data):

    data_train= pd.DataFrame([data_numpy[i] for i in train_index]) # If we don't make "dataframe" and it still is list type, we can't use itertuples below.
    data_test= pd.DataFrame([data_numpy[i] for i in test_index])
    
    train_data_matrix = np.zeros((n_users,n_items))
    
    for line in data_train.itertuples():
        train_data_matrix[line[1]-1,line[2]-1] = line[3]
        
        
    test_data_matrix = np.zeros((n_users,n_items))
    
    for line in data_test.itertuples():
        test_data_matrix[line[1]-1,line[2]-1] = line[3]
        
    
    
    if model == "user":
       
       user_similarity = pairwise_distances(train_data_matrix, metric="correlation") 
       
       user_prediction = predict(train_data_matrix,user_similarity,type="user")
       mean_of_mae  += mean_absolute_error(test_data_matrix, user_prediction)
       error_list.append(mean_absolute_error(test_data_matrix, user_prediction))
       mean_of_mse += mean_squared_error(test_data_matrix, user_prediction)
       error_list_two.append(mean_squared_error(test_data_matrix, user_prediction))
       
       print("user-based mean absolute error: ",mean_absolute_error(test_data_matrix, user_prediction))
       print("user-based mean squared error:",mean_squared_error(test_data_matrix, user_prediction))
       
       
    elif model == "item":

        
        item_similarity = pairwise_distances(train_data_matrix, metric="cosine")
        item_prediction = predict(train_data_matrix,item_similarity,type="item")
        mean_of_mae  += mean_absolute_error(test_data_matrix, item_prediction)
        error_list.append(mean_absolute_error(test_data_matrix, item_prediction))
        mean_of_mse += mean_squared_error(test_data_matrix, item_prediction)
        error_list_two.append(mean_squared_error(test_data_matrix, item_prediction))
        print("item-based mean absolute error: ",mean_absolute_error(test_data_matrix, item_prediction))
        print("item-based mean squared error:",mean_squared_error(test_data_matrix, item_prediction))

    
print("General Mean MAE for all iterations: ", mean_of_mae/fold_number)
print("General Mean MSE for all iterations: ", mean_of_mse/fold_number)
visualize(error_list,error_list_two)