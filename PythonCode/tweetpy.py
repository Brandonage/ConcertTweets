__author__ = 'alvarobrandon'

import tweepy
import pandas as pd
import numpy as np
import time

consumer_key = 'T6eeCmaq96WR2VqjlMhs0AijD'
consumer_secret = 'NGVIjthZ7tExhrwwrRu57yNy2pNMX6TeBqEtAVWTLLiZPpSdn6'
access_token = '232900316-M3MR5wMdGGK1fP0ox8gwVwoQcgBLaRV2kAYwrFkq'
access_token_secret = 'iQSmPRuvjVqmVyBohY8n8Lokase8rF4QcNwo7kKTYDCBF'

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)

auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

states = ['AZ','IL', 'OH', 'WA', 'NY', 'HI','AR', 'TX', 'MI', 'CA', 'PA', 'FL', 'NE', 'UT', 'WI',
                   'MO', 'NC', 'CO', 'NV', 'TN', 'LA', 'VA','DC', 'MD', 'MS', 'NJ', 'CT','OR', 'MA', 'GA',
                  'ID', 'SC','MN', 'WV', 'KY', 'DE','IA','Costa Rica','VT', 'RI',
                  'IN', 'AL', 'NH', 'KS','OK', 'ME','Pennsylvan','WY', 'NM','MT','ND', 'SD','AK']

def nRatingsGreaterThan(n,p):
    g = p.groupby('userid').size()
    indx = g[g>20].index
    p2 = p.loc[p['userid'].isin(indx)]
    return p2



### We set the paths for the different files

pathevents = "/Users/alvarobrandon/GitHub/ConcertTweets/ConcertTweets_v2.5/events.dat"
pathratings = "/Users/alvarobrandon/GitHub/ConcertTweets/ConcertTweets_v2.5/ratings.dat"
pathusers = "/Users/alvarobrandon/GitHub/ConcertTweets/ConcertTweets_v2.5/users.dat"

### Read the data into tables

eventsdf = pd.read_table(pathevents, sep=',', quotechar="\"")
ratingsdf = pd.read_table(pathratings, sep=',', quotechar="\"",encoding="latin-1")
usersdf = pd.read_table(pathusers, sep=',', quotechar="\"",encoding="latin-1")

eventsdf = eventsdf.drop_duplicates("eventId")
## We join the two dataframes by eventId
joindf = pd.merge(ratingsdf,eventsdf, how="inner",on="eventId")
joindf = nRatingsGreaterThan(20,joindf)

joindf = joindf.loc[joindf['state'].isin(states)]


u = joindf["userid"].drop_duplicates()  ## An array with the userId's
u.index = range(0,len(u))


listCommonFriends = []

for user in u:
    while True:
        try:
            friends = np.asarray(api.friends_ids(user))
        except tweepy.TweepError, e:
            print user
            print e
            ##print e.message[0]['code']
            try:
                if e.args[0][0]['code']==88:
                    print('Entering sleep for five minutes')
                    time.sleep(60*5)
                    continue
                else:
                    listCommonFriends.append([])
                    break
            except TypeError, type:
                print('No Code')
                listCommonFriends.append([])
                break
        commonFriends = [id for id in friends if id in u]
        listCommonFriends.append(commonFriends)
        break

listOfUsers = []
for user in u:
    while True:
        try:
            listOfUsers.append(api.get_user(user))
        except tweepy.TweepError, e:
            print user
            print e
            ##print e.message[0]['code']
            try:
                if e.args[0][0]['code']==88:
                    print('Entering sleep for five minutes')
                    time.sleep(60*5)
                    continue
                else:
                    listOfUsers.append('NA')
                    break
            except TypeError, type:
                print('No code')
                listOfUsers.append('NA')
                time.sleep(60*5)
                break
        break

[{u'message': u'User has been suspended.', u'code': 63}]
dict = {}

dict.keys()


public_tweets = api.home_timeline()

for tweet in public_tweets:
    print tweet.text

with open('listOfUsers','rb') as handle:
    listOfUsers = pickle.load(handle) # we load

i=0
for element in listCommonFriends2:
    if element:
        print(u[i], " is following", element)
    i=i+1