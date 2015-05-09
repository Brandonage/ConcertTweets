## We load the libraries needed
library(data.table)
library(Matrix)
require(caret)
library(lsa)

## We print the values that are not 0 from a row in a sparse matrix
sparseRow <- function(r){
  r[which(r!=0)]
}


## Creates a recommendation matrix nrow(test) x ncol(test)
## for each user in test takes the 5 nearest neighbours of m1
## p.dt contains three values
## index row of train, index row of test, cosdistance(train,test)
createRecommendMatrix <- function(train,test,p.dt){
  recMatrix<- Matrix(0,nrow=nrow(test),ncol = ncol(test),sparse = TRUE)
  
  for (i in unique(p.dt$i.y)){ ## I want to go to each row of test that is relevant and get the five nearest neighbours
    print(i)
    ## I get the first five 
    block <- head(p.dt[i.y==i][order(-cosVal)],5)
    
    ## We are goint to weigth the recommendation vector of the user in train with its cosine similarity
    auxMatrix <- Matrix(0,nrow=nrow(block),ncol = ncol(test),sparse=TRUE)
    for (j in 1:nrow(block)){
      ## We retrieve the vector of m1 and multiply it by the cosval
      ## Eg. for user 998567 (0,1,0,0,0,1)*0.4500
      recUser <- as.vector(train[block[j,i.x],]*block[j,cosVal])
      auxMatrix[j,] <- recUser
    }
    auxMatrix.dt <- summary(auxMatrix)
    
    # 5 x 138510 sparse Matrix of class "dgCMatrix", with 18 entries 
    # i      j          x
    # 1  5    236 0.09128709
    # 2  3  17447 0.10206207
    # 3  2  36946 0.11785113
    # 4  4  45848 0.09128709
    auxMatrix.dt <- as.data.table(auxMatrix.dt)
    ## We find the columns that have values <> 0 and we do a sum across the column. We then divide by n
    auxMatrix.dt <- auxMatrix.dt[, sum(x)/nrow(block),by=.(j)]
    ## For the each row of index i we will change the value of the sparse matrix to the sum/n value of the cosine
    for (k in auxMatrix.dt$j){
      recMatrix[i,k] <- auxMatrix.dt[j==k,V1]
    }
  } 
  #cosValues <- apply(pairs,1,calculateCosine)
  rownames(recMatrix) <- rownames(test)
  colnames(recMatrix) <- colnames(test)
  return(recMatrix)
}



## Returns a data.table with the rows of sparse1 that have columns in common with sparse2
relevantRows <- function(sparse1, sparse2){
  summ1<-summary(sparse1)
  summ2 <- summary(sparse2)
  sRes<- merge(summ1, summ2, by="j")
  ## We are going to get the row of sparse1 and row of sparse2 that have a 1 in the same column.
  # head(sRes)
  #   j   i.x x.x  i.y x.y
  # 1 2 11783   1 7806   1
  # 2 2 11783   1 9889   1
  # 3 2 39669   1 7806   1
  # 4 2 39669   1 9889   1
  # 5 3 41554   1 8034   1
  # 6 3 41554   1 7740   1
  
  ## We only keep unique pairs of rows (row from sparse1, row from sparse2)
  pairs <- unique(sRes[c("i.x","i.y")])
  ## We order by the row of sparse2
  pairs<- pairs[order(pairs$i.y),]
  ## We return the pairs as a data.table
  p.dt <- as.data.table(pairs)
  return p.dt
}

accuracy <- function(pred,truth){
  dif <- pred-truth
  t1 <- as.vector(pred)
  t2 <- as.vector(truth)
  dif[which(dif<0)] <- dif[which(dif<0)]*(-1)
  notZero <- dif[unique(c(which(t2!=0),which(t1!=0)))]
  failRatio <- sum(jod)/length(jod)
  return(1-failRatio)
}


## We set the paths of the files that we will use

pathevents = "/Users/alvarobrandon/GitHub/ConcertTweets/ConcertTweets_v2.5/events.dat"
pathratings = "/Users/alvarobrandon/GitHub/ConcertTweets/ConcertTweets_v2.5/ratings.dat"
pathusers = "/Users/alvarobrandon/GitHub/ConcertTweets/ConcertTweets_v2.5/users.dat"

## We read the .dat files into data frames
users.df <- read.table(pathusers,header=TRUE,sep = ',')
events.df <- read.table(pathevents,header=TRUE,sep = ',',fill=TRUE,quote="\"",fileEncoding="latin1")
ratings.df <- read.table(pathratings,header=TRUE,sep = ',',fill=TRUE,quote="\"")

## We are going to transform this data frames into data.tables that will give us a better performance overall
users.dt <- as.data.table(users.df)
events.dt <- as.data.table(events.df)
ratings.dt <- as.data.table(ratings.df)

## We create a list with all the Id's of the users 
u <- users.dt[,userid]
## Another list with all the Id's of the events
e <- events.dt[,eventId]
## A data frame with tuples of (user, concert they went to)
## We won't see duplicate tuples since a user cannot be twice in the same concert
r <- ratings.dt[userid %in% u & eventId %in% e, .(userid,eventId)]


nUsers <- length(u)
nEvents <- length(e)
## We create our sparse matrix (users x events)
m<- Matrix(0,nrow=nUsers,ncol=nEvents,sparse=TRUE)
rownames(m) <- u
colnames(m) <- e



## We change the value in our sparse matrix to one for the pairs of (users,event)
for (i in 1:nrow(r)){
  m[as.character(r[i]$userid),as.character(r[i]$eventId)] <- 1
} 

## We are going to divide the matrix into training and test sets
pos <- rep("A", dim(m)[1])
flds <- createDataPartition(pos, times = 1, p=0.8)
m1 <- m[flds$Resample1,]  ## 49455 x 138510
m2 <- m[-flds$Resample1,] ## 12363 x 138510

## Now that we have two matrices we are going to get the pairs of rows from training and test that have ones in the same concerts/columns. We do so because cosine distance <> 0 only if they share at least one concert. This is for performance purposes

pairs.dt <- relevantRows(s1, s2)


## We calculate the cosine values for each of the rows of m2 with m1
cosValues<- pairs.dt[, cosine(m1[i.x,],m2[i.y,]), by= 1:nrow(pairs.dt)]
pairs.dt$cosVal <- cosValues$V1

# dim(cosValues)
# [1] 122155      2
# > dim(pairs.dt)
# [1] 122155      3

## pairs.dt$i.x is the row index of m1
## pairs.dt$i.y is the row index of m2
# > head(pairs.dt)
#      i.x i.y     cosVal
# 1: 46359   1 0.20000000
# 2: 37803   1 0.42426407
# 3: 39825   1 0.18257419
# 4: 18054   1 0.31622777
# 5:  8436   1 0.08451543
# 6: 14749   1 0.44721360


rMatrix <- createRecommendMatrix(m1,m2,pairs.dt)









# sAcc <- merge(summary(recMatrix),summary(m2),by="i")
# dif <- as.vector(recMatrix[12337,] - m2[12337,])
# acc<- accuracy(recMatrix[12337,],m2[12337,])



for (u in sfusers){
  print(u) 
  print(sparseRow(recMatrix[u,]))
}

recMatrix["19493921",]

[1] "413036079"
8728976    8661585    8793476    8793476    8793478 
0.16329932 0.05773503 0.04714045 0.04714045 0.18652514 

m2["413036079",]




