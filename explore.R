## We define the paths of the .dat files
pathevents = "/Users/alvarobrandon/GitHub/ConcertTweets/ConcertTweets_v2.5/events.dat"
pathratings = "/Users/alvarobrandon/GitHub/ConcertTweets/ConcertTweets_v2.5/ratings.dat"
pathusers = "/Users/alvarobrandon/GitHub/ConcertTweets/ConcertTweets_v2.5/users.dat"
## We load the libraries we need
library(data.table)
library(scales)

## We check the structure of the .dat file

readLines(pathusers,n=5)
# [1] "userid"    "104964154" "26001727"  "77662665"  "99813"

readLines(pathevents,n=5)
# [1] "eventId,eventDate,city,state,latitude,longitude,venue_name,eventURL"                                                                                      
# [2] "8707105,2014-10-07,Philadelphia,PA,39.941495,-75.14777600000001,Legendary Dobbs,http://www.bandsintown.com/event/8707105?artist=bobaflex" 

readLines(pathratings,n=5)
# [1] "userid,band,rating,eventId,venue,eventURL,timestamp"                                                                                                 
# [2] "26159799,straight line stitch,4.0,7456995,The Lizard Lounge,http://www.bandsintown.com/event/7456995?artist=straight+line+stitch,2014-02-01 06:40:14"



##
## We read the data into dataframes

users.df <- read.table(pathusers,header=TRUE,sep = ',')
events.df <- read.table(pathevents,header=TRUE,sep = ',',fill=TRUE,quote="\"",fileEncoding="latin1")
ratings.df <- read.table(pathratings,header=TRUE,sep = ',',fill=TRUE,quote="\"")

users.dt <- as.data.table(users.df)
events.dt <- data.table(events.df,key="eventId")
ratings.dt <- data.table(ratings.df,key="eventId")

## We define a vector with the implicit ratings
implicit <- c("Yes","Maybe")

## We extract only the ratings that are not explicit
irat <- ratings.df[!(ratings.df$rating %in% implicit),]$rating
## We eliminate the factors that don't have values in the subset
irat <- factor(irat)
mean(as.integer(as.character(irat)))
## Count of the ratings
barplot(table(ratings.df$rating))
summary(ratings.df$rating)

## Events in Paris
events.df[(events.df$city=='Paris'),]
ratings.df[(ratings.df$eventId=='9379454'),]


## Can we classify venues?. Some person like concerts in specific venues and it's
## an important factor
venues <- unique(events.df$venue)
venues[!is.na(str_extract(venues,"Park"))]
venues[!is.na(str_extract(venues,"Theater"))]
venues[!is.na(str_extract(venues,"Stadium"))]


#### Data visualisation chapter of the dissertation


joined.dt <- ratings.dt[unique(events.dt)] 

times <- as.character(joined.dt$timestamp)

### we take out the years of the tweets timestamps and save them in an array d
d<-NULL
for (date in times){
  dayAndTime = strsplit(as.character(date)," ")
  day <- dayAndTime[[1]][1]
  split = strsplit(day,"-")
  e <- split[[1]][1]
  d <- c(d,e)
}

### Density of Ratings per year of tweets

joined.dt$year <- as.integer(d)
graph <- ggplot(data= joined.dt,aes(x=year))
graph <- graph + geom_density()

### There is only tweets from 2014 to 2015. What's the count of them?

graph <- ggplot(data= joined.dt,aes(factor(year)))
graph <- graph + geom_bar(aes(fill=factor(year))) + stat_bin(aes(label=..count..), vjust=0, 
        geom="text", position="identity") + xlab("Year") + scale_fill_discrete(name="Year") + theme_bw()

### We are going to save the year and the month of the timestamp to plot the density
d<-NULL
for (date in times){
  dayAndTime = strsplit(as.character(date)," ")
  day <- dayAndTime[[1]][1]
  split = strsplit(day,"-")
  d1 <- split[[1]][1]
  m <- split[[1]][2]
  e <- paste(d1,m,"1",sep = "/")
  d <- c(d,e)
}
## We store this info as a data
d <- as.Date(d)
joined.dt$ratDate <- d


## We plot the density of the ratings by date
graph <- ggplot(data=joined.dt, aes(x=d)) + geom_density(adjust=2.5) + 
  scale_x_date(breaks= date_breaks("month")) + theme_bw() + xlab("Year and Month")
############################
## We are going to do the same but with the years of the concerts/items
################################

times <- as.character(joined.dt$eventDate)

d<-NULL
for (date in times){
  YearMonthDay = strsplit(as.character(date),"-")
  year <- YearMonthDay[[1]][1]
  d <- c(d,year)
}

joined.dt$eventYear <- d

graph <- ggplot(data= joined.dt,aes(factor(eventYear)))
graph <- graph + geom_bar(aes(fill=factor(eventYear))) + stat_bin(aes(label=..count..), vjust=0, 
                                                             geom="text", position="identity") + xlab("Year") + scale_fill_discrete(name="Year") + theme_bw()




### What's the distribution of the ratings

##h <- ggplot(joined.dt,aes(rating,fill=rating)) + geom_bar() + scale_fill_brewer(palette="Paired")
h <- ggplot(joined.dt,aes(rating)) + geom_bar() 





locs <- joined.dt$state
locs.freq.list <- table(locs)

nratings <- data.frame(locs.freq.list)


if (require(maps)) {
  world_map <- map_data("world")
  p<- ggplot(world_map, aes(map_id = region)) +  geom_map(fill="white", map=world_map, color = "black") +
  geom_map(data = nratings, aes(fill = Freq, map_id=locs), map = world_map) + 
    expand_limits(x = world_map$long, y = world_map$lat) + scale_fill_gradient(low="moccasin", high="black") + 
    ggtitle("Number of Ratings by Country")
}

## To do a map of the states and the ratings we are going to change the abbreviate names
## with the full name

## We change the locations to characters
nratings$locs <- as.character(nratings$locs)

## For each location 
for (state in nratings$locs){
  if (nchar(state)==2){   ## If the length of the country is two, it must be a state
    longName=state.name[grep(state,state.abb)]  # grep the longName
    if (length(longName)!=0){
      nratings$locs[which(nratings$locs==state)] <- longName   ### Change it
    }  
  }
}

## Same procedure to plot a map but with the USA map
if (require(maps)) {
  states_map <- map_data("state")
  p<- ggplot(states_map, aes(map_id = region)) +  geom_map(fill="white", map=states_map, color = "black") +
    geom_map(data = nratings, aes(fill = Freq, map_id=tolower(locs)), map = states_map) + 
    expand_limits(x = states_map$long, y = states_map$lat) + scale_fill_gradient(low="moccasin", high="black") + 
    ggtitle("Number of Ratings by State")
}


## We are going to calculate some basic statistics about the data

## First we are going to get the number of ratings by user
ans <- ratings.dt[, .(.N), by = (userid)]
## The maximum number of ratings 
ans[which(ans$N==max(ans$N))]
## mean number of ratings per user
mean(ans$N)

## Now we are going to calculate stats per item
ans <- ratings.dt[, .(.N), by = (eventId)]
min(ans$N)
mean(ans$N)

## Most rated band
ans <- ratings.dt[, .(.N),by=(band)]
ans[order(-N)]

## Number of bands
length(unique(ratings.dt$band))

## Mean ratings and its deviation
## We get all the ratings in an array and we change the Yes to 5 and Maybe to 4
ratings <- ratings.dt$rating
ratings[which(ratings=="Yes")] <- as.factor("5.0")
ratings[which(ratings=="Maybe")] <- as.factor("4.0")
ratings<- as.integer(as.character(ratings))
mean(ratings,na.rm=TRUE)

## What about the countries?. Which ones are the most active ones in the world?
## How many different countries do we have?

ans <- joined.dt[,.(.N),by=(state)]
length(unique(joined.dt$state))
median(ans$N)
ans[N>129,]

#### We plot the results of our different models
### Results for RMSE
method <- rep(c('User','Item','MF'),4)
method <- c(method,'MF')
approach <- rep(c('EventsRatings','BandBinary','BandRatings','ContextRec'),each=3)
approach <- c(approach,'ContextMF')
rmse <- c(4.36,4.51,4.67,3.69,3.99,4.11,3.48,3.75,3.86,3.47,3.68,3.69,2.54)
df <- data.frame(id=seq_along(rmse),approach,method,rmse)
df$approach <- factor(df$approach, levels=unique(as.character(df$approach)) )

ggplot(df, aes(approach, rmse, group = method, colour = method)) +
  geom_path(size=1) + scale_y_continuous(limits=c(2,5)) + labs(title= "RMSE Results for each method")


### Results for TOPN
method <- rep(c('User','Item','MF'),4)
method <- c(method,'MF')
approach <- rep(c('EventsRatings','BandBinary','BandRatings','ContextRec'),each=3)
approach <- c(approach,'ContextMF')
precision <- c(0.0073,0.0041,0.0058,0.0152,0.0074,0.0202,0.0144,0.0073,0.0210,0.0073,0.0030,0.0060,0.0053)
recall <- c(0.1460,0.0832,0.1167,0.3040,0.1484,0.4045,0.2884,0.1464,0.4202,0.1472,0.0611,0.1210,0.1071)
df <- data.frame(id=seq_along(precision),approach,method,precision,recall)
df$approach <- factor(df$approach, levels=unique(as.character(df$approach)) )


ggplot(df, aes(approach, precision, group = method, colour = method)) + geom_path(size=1) + labs(title= "TOPN Precision for each method")

ggplot(df, aes(approach, recall, group = method, colour = method)) + geom_path(size=1) + labs(title= "TOPN Recall for each method")



##### Plotting the results for GRID SEARCH  ######

####    RMSE PERFORMANCE ####
#############################


####  RMSE FOR EventsRatings ######
method <- rep(c('User','Item','MF'),each=4)
KNNAndfactors <- rep(c('(5/30)','(20/50)','(30/75)','(50/100)'),3)
rmse <- c(4.3787, 4.2565, 4.2466, 4.2415, 4.5065, 4.4065, 4.3821, 4.3352, 4.6884, 4.6782, 4.6752, 4.6707)
df <- data.frame(KNNAndfactors,method,rmse)
df$KNNAndfactors <- factor(df$KNNAndfactors, levels=unique(as.character(df$KNNAndfactors)) )

ggplot(df, aes(KNNAndfactors, rmse, group = method, colour = method)) +
  geom_path(size=1) + scale_y_continuous(limits=c(2,5)) + labs(title= "RMSE Results for EventsRatings")

#### RMSE FOR BandsBinary ####

method <- rep(c('User','Item','MF'),each=4)
KNNAndfactors <- rep(c('(5/30)','(20/50)','(30/75)','(50/100)'),3)
rmse <- c(3.7027, 3.1151, 2.9161, 2.6655, 3.9970, 3.7055, 3.5661, 3.3620, 4.1118, 4.1005, 4.0906, 4.0858)
df <- data.frame(KNNAndfactors,method,rmse)
df$KNNAndfactors <- factor(df$KNNAndfactors, levels=unique(as.character(df$KNNAndfactors)) )

ggplot(df, aes(KNNAndfactors, rmse, group = method, colour = method)) +
  geom_path(size=1) + scale_y_continuous(limits=c(2,5)) + labs(title= "RMSE Results for BandsBinary")


#### RMSE FOR BandsRatings ####

method <- rep(c('User','Item','MF'),each=4)
KNNAndfactors <- rep(c('(5/30)','(20/50)','(30/75)','(50/100)'),3)
rmse <- c(3.4666, 2.9092, 2.7232, 2.503, 3.7503, 3.4737, 3.3534, 3.131, 3.8604, 3.8488, 3.8509, 3.8353)
df <- data.frame(KNNAndfactors,method,rmse)
df$KNNAndfactors <- factor(df$KNNAndfactors, levels=unique(as.character(df$KNNAndfactors)) )

ggplot(df, aes(KNNAndfactors, rmse, group = method, colour = method)) +
  geom_path(size=1) + scale_y_continuous(limits=c(2,5)) + labs(title= "RMSE Results for BandsRatings")



#### RMSE FOR ContextRec ####

method <- rep(c('User','Item','MF'),each=4)
KNNAndfactors <- rep(c('1st Iter','2nd Iter','3rd Iter','4th Iter'),3)
rmse <- c(3.343, 2.8303, 2.6516, 2.4331,3.7503, 3.4737, 3.3534, 3.131,3.8604, 3.8488, 3.8509, 3.8353)
df <- data.frame(KNNAndfactors,method,rmse)
df$KNNAndfactors <- factor(df$KNNAndfactors, levels=unique(as.character(df$KNNAndfactors)) )

ggplot(df, aes(KNNAndfactors, rmse, group = method, colour = method)) +
  geom_path(size=1) + scale_y_continuous(limits=c(2,5)) + labs(title= "RMSE Results for ContextRec")

#### RSME for CAMF  ####


Factors <- c(10,30,50,75)
rmse <- c(2.5404, 2.8945, 3.2495, 3.5432)
df <- data.frame(Factors,rmse)


ggplot(df, aes(Factors, rmse)) +
  geom_path(size=1, color='red') + scale_y_continuous(limits=c(2,5)) + scale_x_continuous(limits=c(10,75), breaks=c(10,30,50,75)) +
  labs(title= "RMSE Results for CAMF")



####    TOPN PRECISION PERFORMANCE ####
#######################################


####  TOPN FOR EventsRatings ######
method <- rep(c('User','Item','MF'),each=4)
KNNAndfactors <- rep(c('(5/30)','(20/50)','(30/75)','(50/100)'),3)
prec <- c(0.007, 0.0092, 0.0094, 0.0095, 0.0042, 0.0061, 0.0066, 0.0076, 0.0055, 0.0064, 0.0081, 0.0083 )
df <- data.frame(KNNAndfactors,method,prec)
df$KNNAndfactors <- factor(df$KNNAndfactors, levels=unique(as.character(df$KNNAndfactors)) )

ggplot(df, aes(KNNAndfactors, prec, group = method, colour = method)) +
  geom_path(size=1) +  labs(title= "Precision for EventsRatings")

####  TOPN FOR BandsBinary ######
method <- rep(c('User','Item','MF'),each=4)
KNNAndfactors <- rep(c('(5/30)','(20/50)','(30/75)','(50/100)'),3)
prec <- c(0.0142, 0.0253, 0.02613, 0.0228, 0.0066, 0.0125, 0.014, 0.0148, 0.0208, 0.0218, 0.0219, 0.0216 )
df <- data.frame(KNNAndfactors,method,prec)
df$KNNAndfactors <- factor(df$KNNAndfactors, levels=unique(as.character(df$KNNAndfactors)) )

ggplot(df, aes(KNNAndfactors, prec, group = method, colour = method)) +
  geom_path(size=1) + labs(title= "Precision for BandsBinary")


####  TOPN FOR BandsRatings ######
method <- rep(c('User','Item','MF'),each=4)
KNNAndfactors <- rep(c('(5/30)','(20/50)','(30/75)','(50/100)'),3)
prec <- c(0.0164, 0.0268, 0.0279, 0.0247, 0.0077, 0.012, 0.0129, 0.0146, 0.0216, 0.0216, 0.0223, 0.0226 )
df <- data.frame(KNNAndfactors,method,prec)
df$KNNAndfactors <- factor(df$KNNAndfactors, levels=unique(as.character(df$KNNAndfactors)) )

ggplot(df, aes(KNNAndfactors, prec, group = method, colour = method)) +
  geom_path(size=1) + labs(title= "Precision for BandsRating")


####  TOPN FOR ContextRec ######
method <- rep(c('User','Item','MF'),each=4)
KNNAndfactors <- rep(c('(5/30)','(20/50)','(30/75)','(50/100)'),3)
prec <- c(0.0072, 0.007, 0.006, 0.0047, 0.0077, 0.012, 0.0129, 0.0146, 0.0216, 0.0216, 0.0223, 0.0226 )
df <- data.frame(KNNAndfactors,method,prec)
df$KNNAndfactors <- factor(df$KNNAndfactors, levels=unique(as.character(df$KNNAndfactors)) )

ggplot(df, aes(KNNAndfactors, prec, group = method, colour = method)) +
  geom_path(size=1)  + labs(title= "Precision for ContextRec")

#### TOPN Precision for CAMF  ####


Factors <- c(10,30,50,75)
prec <- c(0.0054, 0.0040, 0.0036, 0.0021)
df <- data.frame(Factors,prec)

ggplot(df, aes(Factors, prec)) +
  geom_path(size=1, color='red') + scale_x_continuous(limits=c(10,75), breaks=c(10,30,50,75)) +
  labs(title= "Prec Results for CAMF")

