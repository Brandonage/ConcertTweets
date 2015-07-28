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


## Function to set to 0's everything under the mean values of the matrix
def smoothFactorisation(rec):
    threshold = rec.mean()
    for i in range(0,rec.shape[0]):
        print(i)
        row = rec[i].tocsc()
        print((row[row>threshold]).shape)
        row[row<threshold] = 0  ## set to 0's
        rec[i]=row ## the new row is the one with the plugged in 0's
    return rec


## eliminate rows that have all 0 values
def eliminateEmptyRows(matrix):
    idx = list(set(matrix.nonzero()[0]))
    return matrix[idx]


## only leaves the rows having more than N ratings in the dataframe P 
def nRatingsGreaterThan(n,p):
    g = p.groupby('userid').size()  ## How many ratings for each userid?
    indx = g[g>n].index ## Index of the ones that have more than n ratings
    p2 = p.loc[p['userid'].isin(indx)] ## We locate rows by index
    return p2


## It creates a Train and Test set from matrix. Percent is the percentage that will go to the test set
def createTrainAndTest(matrix,percent):
    testCols = []
    testRows = []
    limit = matrix.shape[0]
    for i in range(0, limit): ## For each Row
        nonZero = matrix.getrow(i).nonzero()[1] ## columns that are nonZero
        ##if (len(nonZero) > 20): not needed if we do nRatingsGreateThan before
        nsample = (len(nonZero)*percent)/100
        nsample = int(math.ceil(nsample)) ## How many n Ratings is the 20%?
        taken = random.sample(nonZero,nsample) ## We do the sampling
        print(taken)
        testCols.extend(taken) ## We extend the index of the columns
        testRows.extend(([i]*len(taken))) ## The rows are going to be the i we are in at the moment
    ones=np.ones(len(testCols),dtype=np.int) ## The data of the matrix
    testMatrix = csc_matrix((ones,(testRows,testCols)),shape=(matrix.shape[0],matrix.shape[1]))
    trainMatrix = matrix - testMatrix
    return(trainMatrix, testMatrix)


## It creates a Recommendation Matrix using matrix factorisation
def createRecMF(trainMatrix,factors):
    bmf = nimfa.Bmf(trainMatrix, rank=factors, lambda_w=10000000, lambda_h=10000000) ## Decompose the matrix
    fit = bmf()
    rec = (np.dot(fit.basis(),fit.coef())) ## Multiply the two matrices to get the approximation of the original filling the missing values
    rec = smoothFactorisation(rec) ## We set to 0's the values under a threshold
    return(rec)



## It creates a Recommendation Matrix using user collaborative filtering from a training Matrix
def createCosRec(trainMatrix,neighbours):
    spRec = coo_matrix((1,trainMatrix.shape[1])) ## We create an sparse Matrix that we are going to keep filling with rows for each recommendation 
    for i in range(0, trainMatrix.shape[0]):
        nonZeroCols = trainMatrix.getrow(i).nonzero()[1]
        relevantRows = [] ## The rows that share with user U at least one column
        for c in nonZeroCols:
            relevantRows.extend(trainMatrix.getcol(c).nonzero()[0])
        relevantRows= list(set(relevantRows))
        ref=trainMatrix[i] ## The user U to whom we will find the neighbours 
        slicedMatrix = trainMatrix[relevantRows] ## The other users V that share at least one column with U
        distances = distance.cdist(slicedMatrix.toarray(),ref.toarray() , 'cosine')
        block = pd.DataFrame({'rou': relevantRows, 'dist':distances[:,0]})
        block = block[block.rou!=i] ## We discard the neighbour that is the same as U
        block = block[block.dist>1e-12] ## We don't take the ones whose distance is almost 0 because that means they are the same
        block = block.sort('dist',ascending=1) ## The ones with the lower distance first
        topN = block.head(neighbours)
        if not(topN.empty):
            topNrou = topN.iloc[:,1].values
            topNdist = topN.iloc[:,0].values
            print(topNdist,i)
            auxMatrix= trainMatrix[topNrou] ## Our five nearest neighbours
            auxMatrix = auxMatrix.toarray() * (1-topNdist[:,None]) ## We multiply them by the similarity W or (1-dist)
            finalRec = np.sum(auxMatrix,axis=0)/len(topN) ## We sum them by the column and average them for the recommendation
        else:
            finalRec = np.zeros(trainMatrix.shape[1])   
        finalRec[nonZeroCols]=0 ## We don't want to recommend the same concerts a user have been to.
        spRec=vstack([spRec,finalRec]) ## We attach our new found recommendation to the matrix we were building
    spRec = spRec.tocsr()
    spRec = spRec[1:spRec.shape[0]]### remove the first line
    return(spRec)


def createCosRecItem(trainMatrix, neighbours):
    spRec = coo_matrix((trainMatrix.shape[0],1)) ## We create an sparse Matrix that we are going to keep filling with columns for each recommendation
    for i in range(0, trainMatrix.shape[1]):   
        nonZeroRows = trainMatrix.getcol(i).nonzero()[0]
        relevantCols = [] ## The columns that share at least one row with out item i
        for r in nonZeroRows:
            relevantCols.extend(trainMatrix.getrow(r).nonzero()[1])
        relevantCols= list(set(relevantCols))
        ref=trainMatrix[:,i] ## the item I that we want to find neighbours for
        slicedMatrix = trainMatrix[:,relevantCols] ## The other items J that share at least one row
        distances = distance.cdist(slicedMatrix.transpose().toarray(),ref.transpose().toarray() , 'cosine')
        block = pd.DataFrame({'colmns': relevantCols, 'dist':distances[:,0]})
        block = block[block.colmns!=i] ## we discard the same item
        block = block[block.dist>1e-12] ## We don't take the ones close to 0, since it means they are almost the same
        block = block.sort('dist',ascending=1)
        topN = block.head(neighbours)
        if not(topN.empty):
            topNcol = topN.iloc[:,0].values
            topNdist = topN.iloc[:,1].values
            print(topNdist,i)
            auxMatrix= trainMatrix[:,topNcol] ## Our five nearest neighbours
            auxMatrix = auxMatrix.toarray() * (1-topNdist[None,:]) ## We multiply them by the similarity W or (1-dist)
            finalRec = np.sum(auxMatrix,axis=1)/len(topN) ## We sum them by the row and average them for the recommendation
            finalRec = finalRec.reshape(trainMatrix.shape[0],1)  ### we have to reshape it for some reason
        else:
            finalRec = np.zeros(trainMatrix.shape[0])
            finalRec = finalRec.reshape(trainMatrix.shape[0],1)  ### we have to reshape it for some reason
        finalRec[nonZeroRows]=0 ## We don't want to recommend the same concerts a user have been to.
        spRec=hstack([spRec,finalRec]) ## we attach the column
    spRec = spRec.tocsc()
    spRec = spRec[:,1:spRec.shape[1]]### remove the first column
    return(spRec)


## Given a recommendation matrix and its test set it returns precision and recall of that recommendation
def calculatePrecAndRecall(recMatrix,testMatrix):
    totalPrec = []
    totalRecall = []
    for i in range(0,testMatrix.shape[0]):
        rec=recMatrix.getrow(i).nonzero()[1]
        test=testMatrix.getrow(i).nonzero()[1]
        if (rec.size!=0 or test.size!=0):
            tp = [val for val in rec if val in test]
            fp = [val for val in rec if val not in test]
            fn = [val for val in test if val not in rec]
            if (len(tp)+len(fp)==0):
                prec = 0
            else:
                prec = len(tp)/(len(tp)+len(fp))
            if (len(tp)+len(fn)==0):
                recall=0
            else:
                recall = len(tp)/(len(tp)+len(fn))
            print(i)
            print(prec)
            totalPrec.append(prec)
            totalRecall.append(recall)
    return(np.mean(totalPrec),np.mean(totalRecall))

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
    recall = np.true_divide(nhits,len((probeMatrix==5).data))## recall it's the number of hits divided by the number of 5's in the test set
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
pairs = ratingsdf.loc[:,["userid","eventId"]]
## Eliminate the duplicates
pairs = pairs.drop_duplicates()

### We will keep only the ones that have more than 20 ratings by userid

pairs = nRatingsGreaterThan(20,pairs)




## Get the users of the ratings
x = pairs.loc[:,"eventId"]
## Get the columns of the ratings
y = pairs.loc[:,"userid"]

## We store in the array rows= [] the position of the rows inside the matrix that have ones 
rows = []
for value in y:
    ## We compare the value of the data frame with the position in the u array
    rows.append(u[u==value].index[0])
## We do the same for the columns
cols = []
for value in x:
    cols.append(e[e==value].index[0])  ## 7914534-fishbone-baltimore-shindig-festival-2014

## We create a repetition of ones to put inside the matrix
ones=np.ones(len(cols),dtype=np.int)
## We create our final sparse matrix with the ones in the places that the ratings indicate
sp1 = csc_matrix((ones,(rows,cols)),shape=(nUsers,nEvents))
sp1 = eliminateEmptyRows(sp1)
## We eliminate the rows that are empty


## Here we set up the CV with 20 iterations and 80%-20% rule of thumb
cvprecision = []
cvrecall = []

for i in range(0,20):
    
    sptrain, sptest = createTrainAndTest(sp1,20)
    
    
    spRec = createCosRecItem(sptrain)
    
    prec,recall = calculatePrecAndRecall(spRec,sptest)
    


    cvprecision.append(prec)
    cvrecall.append(recall)


## We are going to do Grid search to find new parameters for our training methods

neighboursAndFactors = [(5,30),(20,50),(30,75),(50,100)]


for n, f in neighboursAndFactors:

    userPrec = []
    userRecall = []

    itemPrec = []
    itemRecall = []

    factorsPrec = []
    factorsRecall = []

    sptrain, sptest = createTrainAndTest(sp1,20)
    ### We continue with user CF
    spRec = createCosRec(sptrain,n)
    prec,recall = calculatePrecAndRecall(spRec,sptest)
    userPrec.append(prec)
    userRecall.append(recall)

    ### We continue with item CF
    spRec = createCosRecItem(sptrain,n)
    prec,recall = calculatePrecAndRecall(spRec,sptest)

    itemPrec.append(prec)
    itemRecall.append(recall)

    # ### We start with MF ##############
    # spRec = createRecMF(sptrain,f)
    # (prec,recall) = calculatePrecAndRecall(spRec,sptest)
    # overallPrec = sum(prec)/len(prec)
    # overallRecall = sum(recall)/len(recall)
    #
    #
    # factorsPrec.append(overallPrec)
    # factorsRecall.append(overallRecall)





    
 















