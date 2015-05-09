## We define the paths of the .dat files
pathevents = "/Users/alvarobrandon/GitHub/ConcertTweets/ConcertTweets_v2.5/events.dat"
pathratings = "/Users/alvarobrandon/GitHub/ConcertTweets/ConcertTweets_v2.5/ratings.dat"
pathusers = "/Users/alvarobrandon/GitHub/ConcertTweets/ConcertTweets_v2.5/users.dat"


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
events.dt <- as.data.table(events.df)
ratings.dt <- as.data.table(ratings.df)

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

## What's the average number of ratings per user 
count=0
for (uId in users.df$userid){
  count= count+ length(ratings.df[(ratings.df$userid==uId),]$rating)
  #print(uId)

}
count/length(users.df$userid)

## Can we classify venues?. Some person like concerts in specific venues and it's
## an important factor
venues <- unique(events.df$venue)
venues[!is.na(str_extract(venues,"Park"))]
venues[!is.na(str_extract(venues,"Theater"))]
venues[!is.na(str_extract(venues,"Stadium"))]



