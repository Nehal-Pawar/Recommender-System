# -*- coding: utf-8 -*-

#importing libraries
import numpy as np
import pandas as pd
from surprise import Reader, Dataset, SVD, evaluate
from surprise.model_selection import cross_validate

# Importing the dataset
datafile =np.genfromtxt('train_all_txt.txt',delimiter=' ')
ratings = pd.DataFrame(datafile,columns=[1,2,3])
ratings.columns = ['user', 'item','rating']

#find the given items unique
itemset=set()
missingitemset=set()
for index, row in ratings.iterrows():
       itemset.add(row['item'])
#find the missing items
for i in range(1,1683):
    if i in itemset:
        continue
    else:
        missingitemset.add(i)
#create dataframe for missing items
my_list = list(missingitemset)
df1 = pd.DataFrame( columns=['user','item','rating'])
df1['item'] = pd.DataFrame(np.array(my_list).reshape(32,1), columns = list("a"))
df1['user'] = 943
df1['rating'] = 3#give median ratings

#merge with given data ratings
mergeddata = [df1, ratings]
newinputdata = pd.concat(mergeddata)


#create set for not predicted values
reader = Reader(name=None, line_format='user item rating', sep=None, rating_scale=(1, 5), skip_lines=0)
data = Dataset.load_from_df(newinputdata[['user', 'item', 'rating']], reader)
data.split(n_folds=5)
svd = SVD()
#evaluate(svd, data, measures=['RMSE', 'MAE'])
cross_validate(svd, data, measures=[u'rmse', u'mae'], cv=None, return_train_measures=False, n_jobs=1, pre_dispatch=u'2*n_jobs', verbose=True)
ratingset = data.build_full_trainset()


#train the given set
svd.fit(ratingset)

#get unrated user item list from ratings
unratedtestset = ratingset.build_anti_testset()#dont have ratings


#predict for table for unratedtestset ratings 
finalunratedtestsetpredicted = svd.test(unratedtestset)

#create dataframe and arrange coloumns
readableunratedtestsepredicted = pd.DataFrame(np.array(finalunratedtestsetpredicted), columns = list("abcde"))
#Required columns with new ratings
newtable = readableunratedtestsepredicted.iloc[:,[0,1,3]] 
#rename columns
newtable.columns = ['user', 'item','rating']

#merge dataframes
frames = [newtable, newinputdata]
finaloutputresult = pd.concat(frames)
#sort
finaloutputresult=finaloutputresult.sort_values(['user', 'item'], ascending=[True, True])

#finaloutputresult.dtypes
# convert ratings to display upto 2 decimal
finaloutputresult['rating'] = finaloutputresult['rating'].apply(lambda x: round(x,2))
#convert user and items into integer type  
finaloutputresult['user'] = finaloutputresult['user'].apply(np.int64)
finaloutputresult['item'] = finaloutputresult['item'].apply(np.int64)

#write output to txt file 
finaloutputresult.to_csv('output.txt', header=False, index=False, sep=' ', mode='a')

#END OF PROGRAM