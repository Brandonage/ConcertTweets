__author__ = 'alvarobrandon'


from __future__ import division
import pandas as pd
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
import math
import random
from scipy.spatial import distance
import pickle
import nimfa


# we leave out
# 75226                        1
# QC                           1
# PR                           1
# Connecticu                   1
# Long Island City             1
# Pa, Us                       2
# Brooklyn                     7

states = ['AZ','IL', 'OH', 'WA', 'NY', 'HI','AR', 'TX', 'MI', 'CA', 'PA', 'FL', 'NE', 'UT', 'WI',
                   'MO', 'NC', 'CO', 'NV', 'TN', 'LA', 'VA','DC', 'MD', 'MS', 'NJ', 'CT','OR', 'MA', 'GA',
                  'ID', 'SC','MN', 'WV', 'KY', 'DE','IA','Costa Rica','VT', 'RI',
                  'IN', 'AL', 'NH', 'KS','OK', 'ME','Pennsylvan','WY', 'NM','MT','ND', 'SD','AK']


def matrix_factorization(R, P, Q ,BU, BC,IT, K, steps=1000, alpha=0.0002, beta=0.02):
    Q = Q.T
    er = []
    iteration = []
    for step in xrange(steps):
        print step
        for c in xrange(len(R)):
            tuples = zip(sptrain[0].nonzero()[0],sptrain[0].nonzero()[1])
            for i, j in tuples:
                ##print ("Calculating error")
                eij = R[c][i,j] - np.dot(P[i,:],Q[:,j])-IT[j]-BU[i]-BC[c]
                ##print ("eij:",eij)
                for k in xrange(K):
                    Pderiv = (2 * eij * Q[k][j] - beta * P[i][k])
                    Qderiv = (2 * eij * P[i][k] - beta * Q[k][j])
                    P[i][k] = P[i][k] + alpha * Pderiv
                    ##print ("P[i][k]:",P[i][k])
                    Q[k][j] = Q[k][j] + alpha * Qderiv
                    ##print ("Q[k][j]:",Q[k][j])
                BCderiv = (2 * eij -beta*sum(BC))
                BUderiv = (2*  eij - beta*BU[i])
                BC[c] = BC[c] + alpha * BCderiv
                ##print ("BC[c]:",BC[c] )
                BU[i] = BU[i] + alpha * BUderiv
        # eR = np.dot(P,Q)
        e = 0
        for c in xrange(len(R)):
            tuples = zip(sptrain[0].nonzero()[0],sptrain[0].nonzero()[1])
            for i, j in tuples:
                e = e + pow(R[c][i,j] - np.dot(P[i,:],Q[:,j]) - IT[j] - BU[i] - BC[c]  , 2)
                for k in xrange(K):
                    norm = (pow(P[i][k],2) + pow(Q[k][j],2))
                e = e + (beta/2) * pow(norm + BU[i] + np.sum(BC),2)
        print e
        iteration.append(step)
        er.append(e)
        if e < 0.001:
            break
    return P, Q.T,BU,BC, er

def nRatingsGreaterThan(n,p):
    g = p.groupby('userid').size()
    indx = g[g>20].index
    p2 = p.loc[p['userid'].isin(indx)]
    return p2

def createTrainAndTest(matrix,percent):
    rows = matrix.nonzero()[0]
    cols = matrix.nonzero()[1]
    N = len(rows)
    indices = np.arange(N)
    nsample = (len(indices)*percent)/100
    nsample = int(math.ceil(nsample))
    taken = random.sample(indices,nsample)
    testRows = rows[taken]
    testCols = cols[taken]
    testRatings = matrix[testRows,testCols].tolist()[0]
    testMatrix = csr_matrix((testRatings,(testRows,testCols)),shape=(matrix.shape[0],matrix.shape[1]))
    trainMatrix = matrix - testMatrix
    return(trainMatrix, testMatrix)


def buildMatrixByRegion(ratings,s,u,b):
    pairs = ratings.loc[joindf['state']==s,["userid","band","rating"]]
    pairs.loc[pairs.rating=="Yes",'rating'] = np.float(5.0) ## We change the implicit values
    pairs.loc[pairs.rating=="Maybe",'rating'] = np.float(4.0)
    pairs['rating'] = pairs['rating'].astype(float) ## We change the column of the ratings to float
    g = pairs.groupby(['userid'])
    rows = []
    cols = []
    rat = []
    for name, group in g:   ## name is the userid by which we group before, group is all the band names and its ratings
    ### We are going to group each of the user groups by band to calculate the mean ratings for each band
        g2 = group.loc[:,['band','rating']].groupby('band')
        meanRatings = g2['rating'].mean()
        z  = list(group['band']) ## A list with the bands that the user have been to. The names can be repeated
        d = Counter(z)  ## A dictionary with the distinct names of bands and the number of occurences of each one
        for band, count in d.iteritems(): ## Eg. band "Arctic monkeys" count = 3
            cols.append(b[b==band].index[0]) ## We append the position of the band in the matrix
            freq = len(list(c for c in d.itervalues() if c <= count)) ## The number of bands which count <= (current band count)
            r = (meanRatings[band] * freq)/len(d) ## We do this in a scale [0,5]
            rat.append(r)
    ### name is the user
        userNo = (u[u==name].index[0])
        rows.extend([userNo]*len(d)) ## We extend with the row position of the user repeated n times where n is the number of columns
    sp1b = csr_matrix((map(float,rat),(rows,cols)),shape=(len(u),len(b)))
    return(sp1b)


def calculateBundleTopN(train, test, Rec):
    tn = []
    for i in xrange(len(Rec)):
        if np.any((test[i]==5).nonzero()):
            tn.append(topN(train[i],test[i], Rec[i]))
    prec = []
    recall = []
    for e in tn:
        prec.append(e[0])
        recall.append(e[1])
    p = np.sum(prec)/len(prec)
    r = np.sum(recall)/len(recall)
    return(p,r)



def topN(trainMatrix,probeMatrix, recMatrix):
    nhits = 0
    for i in xrange(probeMatrix.shape[0]): ## For each user
        fives = (probeMatrix[i]==5).nonzero()[1]  ## We take the items positions rated with fives
        unrated = (trainMatrix[i]==0).nonzero()[1]   ## We take all the unrated items of that user
        for fpos in fives:  ### for each five that the user has
            if len(fives)>0:
                if (len(unrated) > 1000):
                    nsample = 1000
                else:
                    nsample = len(unrated)
                taken = random.sample(unrated,nsample) ## We randomly select 1000 items that have been unrated
                taken.append(fpos) ## We also include the item that is five in the test set to see its value in the recommendation
                taken.sort() ## we sort the rows taken
                taken = np.array(taken) ## we need to change it to an array so we can use (recMatrix[i,taken]<>0) as index
                ##print (recMatrix[i,taken]<>0).nonzero()[1]
                print i
                print taken
                print fpos
                tuples = zip(recMatrix[i,taken],taken)  ## we do tuples (rating, item)
                tuples = sorted(tuples,key=lambda x:x[0],reverse=True) ## We sort by rating
                if (fpos in [e[1] for e in tuples[0:19]]): ### If the item that was five it's in the top 20
                    nhits+=1 ## we have a hit
    recall = np.true_divide(nhits,len((probeMatrix==5).data)) ## recall it's the number of hits divided by the number of 5's in the test set
    prec = recall/20
    print ("nhits",nhits)
    print ("ntest",len((probeMatrix==5).data))
    return (prec, recall)


## Calculate the RMSE given a recommendation matrix and a test matrix
def calculateBundleRMSE(recBundle,testBundle):
    pred = []
    test = []
    for i in range(0,len(recBundle)):
        pred.extend(recBundle[i][testBundle[i].nonzero()[0],testBundle[i].nonzero()[1]].tolist()) ## Put in an array all the values from the recommendation matrix that are in the same pos as the test matrix
        test.extend(testBundle[i][testBundle[i].nonzero()[0],testBundle[i].nonzero()[1]].tolist()[0]) ## Put in an array all the values from the test matrix
    p = np.matrix(pred)
    t = np.matrix(test)
    rmse=np.sum(np.square(p-t))/t.shape[1] ## The formula to calculate RMSE
    return(np.sqrt(rmse))






### We set the paths for the different files

pathevents = "/Users/alvarobrandon/GitHub/ConcertTweets/ConcertTweets_v2.5/events.dat"
pathratings = "/Users/alvarobrandon/GitHub/ConcertTweets/ConcertTweets_v2.5/ratings.dat"
pathusers = "/Users/alvarobrandon/GitHub/ConcertTweets/ConcertTweets_v2.5/users.dat"

### Read the data into tables

eventsdf = pd.read_table(pathevents, sep=',', quotechar="\"")
ratingsdf = pd.read_table(pathratings, sep=',', quotechar="\"",encoding="latin-1")
usersdf = pd.read_table(pathusers, sep=',', quotechar="\"",encoding="latin-1")

## We drop the duplicate Id's of events since we don't need them to have the basic
## contextual info about the event
eventsdf = eventsdf.drop_duplicates("eventId")
## We join the two dataframes by eventId
joindf = pd.merge(ratingsdf,eventsdf, how="inner",on="eventId")
joindf = nRatingsGreaterThan(20,joindf)

joindf = joindf.loc[joindf['state'].isin(states)]

u = joindf["userid"].drop_duplicates()  ## An array with the userId's
u.index = range(0,len(u))
b = joindf['band'].drop_duplicates() ## An array with the names of the bands
b.index = range(0,len(b)) ## We change the index of the array so each band has an unique umber





factors = [30,50,75]
resultsRMSE = []
resultsPrec = []
resultsRecall = []

for K in factors:
    sptrain = []
    sptest = []
    for s in states:
        spSub = buildMatrixByRegion(joindf,s,u,b)
        spSubTrain, spSubTest = createTrainAndTest(spSub,20)
        sptrain.append(spSubTrain)
        sptest.append(spSubTest)

    N = sptrain[0].shape[0] ### Number of users
    M = sptrain[0].shape[1] ### Number of items

    IT = []
    for j in xrange(M):
        itemMeans = [0] ### The mean of the items is 0 in the first iteration
        for c in xrange(len(sptrain)):
            r = sptrain[c][:,j].data.mean()
            if (not math.isnan(r)):
                itemMeans.append(r)
        IT.append(np.mean(itemMeans))
    P = np.random.rand(N,K)
    Q = np.random.rand(M,K)
    BU = np.random.rand(N)
    BC = np.random.rand(len(sptrain))

    nP, nQ, nBU, nBC, error = matrix_factorization(sptrain, P, Q,BU,BC,IT, K)
    finalBU = np.tile(nBU,(sptrain[0].shape[1],1)).T
    finalBC=[]
    for i in xrange(len(nBC)):
        finalBC.append(np.tile(nBC[i],(sptrain[0].shape[0],sptrain[0].shape[1])))

    finalIT = np.tile(IT,(sptrain[0].shape[0],1))

    finalRec = []
    for c in xrange(len(sptrain)):
        finalRec.append(np.dot(nP,nQ.T)+finalIT+finalBU+finalBC[c])

    prec, recall = calculateBundleTopN(sptrain,sptest,finalRec)
    rmse = calculateBundleRMSE(finalRec,sptest)
    resultsRMSE.append(rmse)
    resultsPrec.append(prec)
    resultsRecall.append(recall)









####################################################################



ratings=[]
for i in range(0,5):
    data = np.random.randint(6,size=15)
    rows = np.random.randint(50,size=15)
    cols = np.random.randint(50,size=15)
    m = csr_matrix((data,(rows,cols)),shape=(50,50))
    ratings.append(m)




N = ratings[0].shape[0] ### Number of users
M = ratings[0].shape[1] ### Number of items
K = 2

IT = []
for j in xrange(M):
    itemMeans = [0] ### The mean of the items is 0 in the first place
    for c in xrange(len(ratings)):
        r = ratings[c][:,j].data.mean()
        if (not math.isnan(r)):
            itemMeans.append(r)
    IT.append(np.mean(itemMeans))


P = np.random.rand(N,K)
Q = np.random.rand(M,K)
BU = np.random.rand(N)
BC = np.random.rand(len(ratings))




nP, nQ, nBU, nBC = matrix_factorization(ratings, P, Q,BU,BC,IT, K)

BU = np.tile(nBU,(ratings[0].shape[0],1)).T
BC=[]
for i in xrange(len(nBC)):
    BC.append(np.tile(nBC[i],(ratings[0].shape[0],ratings[0].shape[1])))


IT = np.tile(IT,(ratings[0].shape[1],1))

finalRec = []
for c in xrange(len(ratings)):
    finalRec.append(np.dot(nP,nQ.T)-IT-BU-BC[c])
