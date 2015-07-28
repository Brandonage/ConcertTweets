__author__ = 'alvarobrandon'
from __future__ import division
import pandas as pd
import numpy as np
from scipy.sparse import csc_matrix, coo_matrix, csr_matrix, vstack, hstack
import math
import random
from scipy.spatial import distance
import pickle
import nimfa



##### CHANGES TO BE MADE  ######
##### 1. CONSIDER CHANGING THE TYPES OF SPARSE MATRIX (CSC, CSR) TO INCREASE PERFORMANCE  #####

## Given top 5 neighbour rows t5, their distances tdist, the training matrix t to be able to extract the data
## and axis = 0 = by row or axis = 1 = by column this function builds the vector W 
## that we will use to divide the weighted vectors for a recommendation
def buildVector(t5,tdist,t,axis):
    ## if we are building a row vector W
    if axis==0:
        weight = np.zeros((len(t5),t.shape[1])) ## We create a matrix of 5 x number of events
        i = 0
        for rou in t5:
            nonZeroCols=t.getrow(rou).nonzero()[1] ## for each column that is not 0 in the row i
            weight[i,nonZeroCols]=1-tdist[i] ## we set the non zero values to the similarity value W
            i += 1
        return(np.sum(weight,axis=0)) ## We return the sum by the row
    else:
        weight = np.zeros((t.shape[0],len(t5))) ## We create a matrix of number of events x 5
        i=0
        for col in t5:
            nonZeroRows=t.getcol(col).nonzero()[0] ## for each row that is not 0 in the column i
            weight[nonZeroRows,i]=1-tdist[i] ## we set the non zero values to the similarity value W
            i +=1
        return(np.sum(weight,axis=1)) ## We return the sum by the row

## eliminate rows that have all 0 values
def eliminateEmptyRows(matrix):
    idx = list(set(matrix.nonzero()[0]))
    return matrix[idx]

## only leaves the rows having more than N ratings in the dataframe P 
def nRatingsGreaterThan(n,p):
    g = p.groupby('userid').size()
    indx = g[g>20].index
    p2 = p.loc[p['userid'].isin(indx)]
    return p2


## It creates a Train and Test set from matrix. Percent is the percentage that will go to the test set
def createTrainAndTest(matrix,percent):
    testCols = []
    testRows = []
    testRatings = []
    limit = matrix.shape[0]
    for i in range(0, limit): ## for each row
        nonZero = matrix.getrow(i).nonzero()[1] ## which columns are not zero
        ## if (len(nonZero) > 20): not needed if we do nRatingsGreateThan before
        nsample = (len(nonZero)*percent)/100
        nsample = int(math.ceil(nsample)) ## How many ratings are 20%
        taken = random.sample(nonZero,nsample) ## We do the sampling
        taken.sort() ## we sort the rows taken
        print(taken)
        testCols.extend(taken) ## we extend the columns taken
        testRows.extend(([i]*len(taken))) ## and the rows
        r = matrix[i,taken].data
        testRatings.extend(r.tolist())### The ratings of matrix[i,testCols])
    print len(testRatings)
    print len(testRows)
    print len(testCols)
    testMatrix = csr_matrix((testRatings,(testRows,testCols)),shape=(matrix.shape[0],matrix.shape[1]))
    trainMatrix = matrix - testMatrix
    return(trainMatrix, testMatrix)

## It creates a Recommendation Matrix using matrix factorisation
def createRecMF(trainMatrix,factors):
    nmf = nimfa.Nmf(trainMatrix, rank=factors, max_iter=100) ## Decompose the matrix
    fit = nmf()
    rec = (np.dot(fit.basis(),fit.coef())) ## Multiply the two matrices to get the approximation of the original filling the missing values
    return(rec)

## It creates a Recommendation Matrix using user collaborative filtering from a training Matrix
def createCosRec(trainMatrix,neighbours):
    spRec = coo_matrix((1,trainMatrix.shape[1])) ## We create an sparse Matrix that we are going to keep filling with rows for each recommendation 
    ##spRec = coo_matrix((nUsers,nEvents))
    for i in range(0, trainMatrix.shape[0]):   
        nonZeroCols = trainMatrix.getrow(i).nonzero()[1]
        relevantRows = [] ## The rows that share with user U at least one column
        for c in nonZeroCols:
            relevantRows.extend(trainMatrix.getcol(c).nonzero()[0])
        relevantRows= list(set(relevantRows))
        ref=trainMatrix[i] ## The user U to whom we will find the neighbours
        slicedMatrix = trainMatrix[relevantRows] ## The other users V that share at least one column with U
        distances = distance.cdist(slicedMatrix.toarray(),ref.toarray() , 'correlation')
        block = pd.DataFrame({'rou': relevantRows, 'dist':distances[:,0]})
        block = block[block.rou!=i] ## We discard the neighbour that is the same as U
        block = block[block.dist<1] ## We don't want users that have a negative correlation 
        block = block[block.dist>1e-12]  ## We don't take the ones whose distance is almost 0 because that means they are the same
        block = block.sort('dist',ascending=1) ## The ones with the lower distance first
        topN = block.head(neighbours)
        if not(topN.empty):
            topNrou = topN.iloc[:,1].values
            topNdist = topN.iloc[:,0].values
            print(topNdist,i)
            auxMatrix= trainMatrix[topNrou] ## Our five nearest neighbours
            auxMatrix = auxMatrix.toarray() * (1-topNdist[:,None]) ## We multiply them by the similarity W or (1-dist)
            w = buildVector(topNrou,topNdist,trainMatrix,axis=0)
            finalRec = np.sum(auxMatrix,axis=0)/w ## We sum them by the column and divide them by the w vector
            finalRec[np.isnan(finalRec)]=0 ## we set to 0's the ones that have nan values after dividing by the W
        else:
            finalRec = np.zeros(trainMatrix.shape[1])   
        finalRec[nonZeroCols]=0 ## We don't want to recommend the same concerts a user have been to.
        spRec=vstack([spRec,finalRec]) ## We attach our new found recommendation to the matrix we were building
    spRec = spRec.tocsr()
    spRec = spRec[1:spRec.shape[0]]### remove the first line
    return(spRec)


def createCosRecItem(trainMatrix,neighbours):
    spRec = coo_matrix((trainMatrix.shape[0],1)) ## We create an sparse Matrix that we are going to keep filling with columns for each recommendation
    ##spRec = coo_matrix((nUsers,nEvents))
    for i in range(0, trainMatrix.shape[1]):   
        nonZeroRows = trainMatrix.getcol(i).nonzero()[0]
        relevantCols = [] ## The columns that share at least one row with out item i
        for r in nonZeroRows:
            relevantCols.extend(trainMatrix.getrow(r).nonzero()[1])
        relevantCols= list(set(relevantCols))
        ref=trainMatrix[:,i] ## the item I that we want to find neighbours for
        slicedMatrix = trainMatrix[:,relevantCols] ## The other items J that share at least one row
        distances = distance.cdist(slicedMatrix.transpose().toarray(),ref.transpose().toarray() , 'correlation')
        block = pd.DataFrame({'colmns': relevantCols, 'dist':distances[:,0]})
        block = block[block.colmns!=i]  ## we discard the same item
        block = block[block.dist<1] ## We don't want users that have a negative correlation
        block = block[block.dist>1e-12] ## We don't take the ones whose distance is almost 0 because that means they are the same
        block = block.sort('dist',ascending=1) ## The ones with the lower distance first
        topN = block.head(neighbours)
        if not(topN.empty):
            topNcol = topN.iloc[:,0].values
            topNdist = topN.iloc[:,1].values
            print(topNdist,i)
            auxMatrix= trainMatrix[:,topNcol] ## Our five nearest neighbours
            auxMatrix = auxMatrix.toarray() * (1-topNdist[None,:]) ## We multiply them by the similarity W or (1-dist)
            w = buildVector(topNcol,topNdist,trainMatrix,axis=1) ## We create our vector W
            finalRec = np.sum(auxMatrix,axis=1)/w ## Divide the sum by the vector W
            finalRec[np.isnan(finalRec)]=0
            finalRec = finalRec.reshape(trainMatrix.shape[0],1)  ### we have to reshape it for some reason
        else:
            finalRec = np.zeros(trainMatrix.shape[0])
            finalRec = finalRec.reshape(trainMatrix.shape[0],1)  ### we have to reshape it for some reason
        finalRec[nonZeroRows]=0 ## We don't want to recommend the same concerts a user has been to.
        spRec=hstack([spRec,finalRec])
    spRec = spRec.tocsc()
    spRec = spRec[:,1:spRec.shape[1]]### remove the first column
    return(spRec)


## Calculate the RMSE given a recommendation matrix and a test matrix
def calculateRMSE(recMatrix,testMatrix):
    pred = recMatrix[testMatrix.nonzero()[0],testMatrix.nonzero()[1]] ## Put in an array all the values from the recommendation matrix that are in the same pos as the test matrix
    test = testMatrix[testMatrix.nonzero()[0],testMatrix.nonzero()[1]] ## Put in an array all the values from the test matrix
    rmse=np.sum(np.square(pred-test))/len(testMatrix.data) ## The formula to calculate RMSE
    return(np.sqrt(rmse))


def topN(trainMatrix,probeMatrix, recMatrix):
    nhits = 0
    for i in xrange(probeMatrix.shape[0]): ## For each user
        fives = (probeMatrix[i]==5).nonzero()[1]  ## We take the items positions rated with fives
        unrated = (trainMatrix[i]==0).nonzero()[1]   ## We take all the unrated items of that user
        for fpos in fives:  ### for each five that the user has
            nsample = 1000
            taken = random.sample(unrated,nsample) ## We randomly select 1000 items that have been unrated
            taken.append(fpos) ## We also include the item that is five in the test set to see its value in the recommendation
            taken.sort() ## we sort the rows taken
            taken = np.array(taken) ## we need to change it to an array so we can use (recMatrix[i,taken]<>0) as index
            nonZero = taken[(recMatrix[i,taken]<>0).nonzero()[1]]  ### which of this items are not 0??
            tuples = zip(recMatrix[i,nonZero].toarray()[0],nonZero)  ## we do tuples (rating, item)
            tuples = sorted(tuples,key=lambda x:x[0],reverse=True) ## We sort by rating
            if (fpos in [e[1] for e in tuples[0:19]]): ### If the item that was five it's in the top 20
                nhits+=1 ## we have a hit
    recall = np.true_divide(nhits,len((probeMatrix==5).data)) ## recall it's the number of hits divided by the number of 5's in the test set
    prec = recall/20
    return (prec, recall)











### We set the paths for the different files

pathevents = "/Users/alvarobrandon/GitHub/ConcertTweets/ConcertTweets_v2.5/events.dat"
pathratings = "/Users/alvarobrandon/GitHub/ConcertTweets/ConcertTweets_v2.5/ratings.dat"
pathusers = "/Users/alvarobrandon/GitHub/ConcertTweets/ConcertTweets_v2.5/users.dat"

### Read the data into tables

eventsdf = pd.read_table(pathevents, sep=',', quotechar="\"")
ratingsdf = pd.read_table(pathratings, sep=',', quotechar="\"",encoding="latin-1")
usersdf = pd.read_table(pathusers, sep=',', quotechar="\"",encoding="latin-1")


## Read the userId's into an array
u = usersdf["userid"]
## Read the eventId's into an array and eliminate duplicates
e = eventsdf["eventId"]


### Create two variables with the dimensions (nUsersxnEvents)
nUsers = len(u)
nEvents = len(e)

## Go through the ratings and collect the pairs of user and event
pairs = ratingsdf.loc[:,["userid","eventId","rating"]]
## Eliminate the duplicates
pairs = pairs.drop_duplicates(subset=["userid","eventId"])
## Some events ID's have strange symbols at the end so we eliminate those, and we change the type of them

##pairs = pairs[-(pairs.eventId.map(len) > 7)]
##pairs['eventId'] = pairs['eventId'].astype(int)



## We are going to change the implicit ratings for explicit ones. yes -> 5, maybe -> 4

pairs.loc[pairs.rating=="Yes",'rating'] = np.float(5.0)
pairs.loc[pairs.rating=="Maybe",'rating'] = np.float(4.0)





### We will keep only the ones that have more than 20 ratings by userid

pairs = nRatingsGreaterThan(20,pairs)





x = pairs.loc[:,"eventId"]## Get the columns of the ratings
y = pairs.loc[:,"userid"]## Get the users of the ratings
rat = pairs.loc[:,"rating"]
rows = []   ## We store in the array rows= [] the position of the rows inside the matrix that have ones
for value in y:
    print(value)
    rows.append(u[u==value].index[0])## We compare the value of the data frame with the position in the u array
cols = []   ## We do the same for the columns
for value in x:
    print(value)
    cols.append(e[e==value].index[0])  ## 7914534-fishbone-baltimore-shindig-festival-2014


## We create our final sparse matrix with the ratings in the places that columns and rows indicate
sp1 = csr_matrix((map(float,rat),(rows,cols)),shape=(nUsers,nEvents))
sp1 = eliminateEmptyRows(sp1)

cvrmse = []


for i in range(0,20):
    
    sptrain, sptest = createTrainAndTest(sp1,20)
    
    ##### calculating the cosine distance from one vector to a matrix ####
    
    spRec = createRecMF(sptrain,30)
    
    prec, recall = topN(sptrain,sptest, spRec)

    rmse = calculateRMSE(spRec,sptest)
    

    cvrmse.append(rmse)
    
 



neighboursAndFactors = [(5,30),(20,50),(30,75),(50,100)]
userRMSE = []
userPrec = []
userRecall = []
itemRMSE = []
itemPrec = []
itemRecall = []
factorsRMSE = []
factorsPrec = []
factorsRecall = []
sptrain, sptest = createTrainAndTest(sp1,20)
for n, f in neighboursAndFactors:
    ### We start with user CF
    spRec = createCosRec(sptrain,n)
    prec, recall = topN(sptrain,sptest, spRec)
    rmse = calculateRMSE(spRec,sptest)
    userRMSE.append(rmse)
    userPrec.append(prec)
    userRecall.append(recall)
    ### We continue with item CF
    spRec = createCosRecItem(sptrain,n)
    prec, recall = topN(sptrain,sptest, spRec)
    rmse = calculateRMSE(spRec,sptest)
    itemRMSE.append(rmse)
    itemPrec.append(prec)
    itemRecall.append(recall)
    ### We continue with MF ##############
    spRec = createRecMF(sptrain,f)
    prec, recall = topN(sptrain,sptest, spRec)
    rmse = calculateRMSE(spRec,sptest)
    factorsRMSE.append(rmse)
    factorsPrec.append(prec)
    factorsRecall.append(recall)

results = []
for i in xrange(len(neighboursAndFactors)):
    results.append(round(np.mean(userRMSE[i]),4))









# (array([  2.22044605e-16,   2.92893219e-01,   2.92893219e-01,
#          2.92893219e-01,   2.92893219e-01]), 3243)
#
# lsnmf = nimfa.Lsnmf(sptrain, seed='random_vcol', rank=30, max_iter=100)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
with open('listCommonFriends2','wb') as handle:
    pickle.dump(listCommonFriends2,handle) # we pickle the graph into graph
#
# with open('sp1rat','rb') as handle:
#     sp1 = pickle.load(handle) # we load
#
# with open('cols','rb') as handle:
#     cols = pickle.load(handle) # we load
#
# totalPrec = []
# totalRecall = []
# for i in range(0,nUsers):
#     rec=spRec.getrow(i).nonzero()[1]
#     test=sptest.getrow(i).nonzero()[1]
#     tp = [val for val in rec if val in test]
#     fp = [val for val in rec if val not in test]
#     fn = [val for val in test if val not in rec]
#     prec = len(tp)/(len(tp)+len(fp))
#     recall = len(tp)/(len(tp)+len(fn))
#     totalPrec.append(prec)
#     totalRecall.append(recall)
#
#
# 5235, 70613, 74421
#
# ratingsdf[ratingsdf.eventId==e[70613]]
#
# [i for i, e in enumerate(prec) if e!=0]
#
#
#
#


