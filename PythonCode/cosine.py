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


def nRatingsGreaterThan(n,p):
    g = p.groupby('userid').size()
    indx = g[g>20].index
    p2 = p.loc[p['userid'].isin(indx)]
    return p2



def createTrainAndTest(matrix,percent):

    testCols = []
    testRows = []
    

    limit = matrix.shape[0]

    for i in range(0, limit): 

        nonZero = matrix.getrow(i).nonzero()[1]

        if (len(nonZero) > 20):
            nsample = (len(nonZero)*percent)/100
            nsample = int(math.ceil(nsample))
            taken = random.sample(nonZero,nsample)
            print(taken)
            testCols.extend(taken)
            testRows.extend(([i]*len(taken)))
    
    ones=np.ones(len(testCols),dtype=np.int)
    testMatrix = csr_matrix((ones,(testRows,testCols)),shape=(matrix.shape[0],matrix.shape[1]))
    trainMatrix = matrix - testMatrix
    return(trainMatrix, testMatrix)

def createCosRec(trainMatrix):

    spRec = coo_matrix((1,trainMatrix.shape[1]))  
    ##spRec = coo_matrix((nUsers,nEvents))
    
    for i in range(0, trainMatrix.shape[0]):   
        nonZeroCols = trainMatrix.getrow(i).nonzero()[1]
        relevantRows = []
        for c in nonZeroCols:
            relevantRows.extend(trainMatrix.getcol(c).nonzero()[0])
        relevantRows= list(set(relevantRows))
        ref=trainMatrix[i]
        slicedMatrix = trainMatrix[relevantRows]
        distances = distance.cdist(slicedMatrix.toarray(),ref.toarray() , 'cosine')
        block = pd.DataFrame({'rou': relevantRows, 'dist':distances[:,0]})
        block = block[block.rou!=i]
        block = block[block.dist!=0]
        block = block.sort('dist',ascending=1)
        top5 = block.head(5)
        if not(top5.empty):
            top5rou = top5.iloc[:,1].values
            top5dist = top5.iloc[:,0].values
            print(top5dist,i)
            auxMatrix= trainMatrix[top5rou]
            auxMatrix = auxMatrix.toarray() * top5dist[:,None]
            finalRec = np.sum(auxMatrix,axis=0)/len(top5)
        else:
            finalRec = np.zeros(trainMatrix.shape[1])   
        finalRec[nonZeroCols]=0 ## We don't want to recommend the same concerts a user have been to.
        spRec=vstack([spRec,finalRec])
    spRec = spRec.tocsr()
    spRec = spRec[1:spRec.shape[0]]### remove the first line
    return(spRec)


def createCosRecItem(trainMatrix):

    spRec = coo_matrix((trainMatrix.shape[0],1)) 
    ##spRec = coo_matrix((nUsers,nEvents))
    
    for i in range(0, trainMatrix.shape[1]):   
        nonZeroRows = trainMatrix.getcol(i).nonzero()[0]
        relevantCols = []
        for r in nonZeroRows:
            relevantCols.extend(trainMatrix.getrow(r).nonzero()[1])
        relevantCols= list(set(relevantCols))
        ref=trainMatrix[:,i]
        slicedMatrix = trainMatrix[:,relevantCols]
        distances = distance.cdist(slicedMatrix.transpose().toarray(),ref.transpose().toarray() , 'cosine')
        block = pd.DataFrame({'colmns': relevantCols, 'dist':distances[:,0]})
        block = block[block.colmns!=i]
        block = block[block.dist!=0]
        block = block.sort('dist',ascending=1)
        top5 = block.head(5)
        if not(top5.empty):
            top5col = top5.iloc[:,0].values
            top5dist = top5.iloc[:,1].values
            print(top5dist,i)
            auxMatrix= trainMatrix[:,top5col]
            auxMatrix = auxMatrix.toarray() * top5dist[None,:]
            finalRec = np.sum(auxMatrix,axis=1)/len(top5)
            finalRec = finalRec.reshape(trainMatrix.shape[0],1)  ### we have to reshape it for some reason
        else:
            finalRec = np.zeros(trainMatrix.shape[0])
            finalRec = finalRec.reshape(trainMatrix.shape[0],1)  ### we have to reshape it for some reason
        finalRec[nonZeroRows]=0 ## We don't want to recommend the same concerts a user have been to.
        spRec=hstack([spRec,finalRec])
    spRec = spRec.tocsc()
    spRec = spRec[:,1:spRec.shape[1]]### remove the first column
    return(spRec)



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
    return(totalPrec,totalRecall)







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
    print(value)
    ## We compare the value of the data frame with the position in the u array
    rows.append(u[u==value].index[0])
## We do the same for the columns
cols = []
for value in x:
    print(value)
    cols.append(e[e==value].index[0])  ## 7914534-fishbone-baltimore-shindig-festival-2014

## We create a repetition of ones to put inside the matrix
ones=np.ones(len(cols),dtype=np.int)
## We create our final sparse matrix with the ones in the places that the ratings indicate
sp1 = csr_matrix((ones,(rows,cols)),shape=(nUsers,nEvents))

####### We do the sampling of the columns to construct another test sparse matrix #####

cvprecision = []
cvrecall = []

for i in range(0,20):
    
    sptrain, sptest = createTrainAndTest(sp1,20)
    
    ##### calculating the cosine distance from one vector to a matrix ####
    
    spRec = createCosRec(sptrain)
    
    (prec,recall) = calculatePrecAndRecall(spRec,sptest)
    
    overallPrec = sum(prec)/len(prec)
    overallRecall = sum(recall)/len(recall)

    cvprecision.append(overallPrec)
    cvrecall.append(overallRecall)
    
 


(array([  2.22044605e-16,   2.92893219e-01,   2.92893219e-01,
         2.92893219e-01,   2.92893219e-01]), 3243)

















with open('sp1','wb') as handle:
        pickle.dump(sp1,handle) # we pickle the graph into graph

with open('sp1','rb') as handle:
    sp1 = pickle.load(handle) # we load

with open('cols','rb') as handle:
    cols = pickle.load(handle) # we load

totalPrec = []
totalRecall = []
for i in range(0,nUsers):
    rec=spRec.getrow(i).nonzero()[1]
    test=sptest.getrow(i).nonzero()[1]
    tp = [val for val in rec if val in test]
    fp = [val for val in rec if val not in test]
    fn = [val for val in test if val not in rec]
    prec = len(tp)/(len(tp)+len(fp))
    recall = len(tp)/(len(tp)+len(fn))
    totalPrec.append(prec)
    totalRecall.append(recall)


5235, 70613, 74421

ratingsdf[ratingsdf.eventId==e[70613]]

[i for i, e in enumerate(prec) if e!=0]






