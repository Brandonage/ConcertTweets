# ConcertTweets
Repository for the analysis of the ConcertTweets dataset and design of a recommender system.
This dataset was extracted using a Twitter streaming API, recollecting ratings given by users to concerts. Every time a user rates one gig in the BandsInTown app, a tweet can be automatically submitted, expressing the rating or the interest in going to this event. The dataset contains information on 250.000 ratings given by 61.803 users on 116.344 different events with 23.861 different bands. The ratings are both explicit (i.e ratings from 0..5) and implicit (i.e ratings indicating interest in the concert {‘Yes’, ‘Maybe’, ‘No’}.

The objective of the system is to recommend new bands and events to users, based on what they liked before. This is achieved by finding groups of users that have a common taste for music, like for example "country rock". In this way, if user A and B are in the same group or cluster, we can suggest to user A bands that user B liked. This type of systems are an important part of internet services like Amazon, Pandora or Netflix, where we have endless items and possibilities to choose or consume.
All of the experiments were implemented with Python and the help of the numpy, pandas, scipy, and nimfa libraries [5][7]. This libraries allow us to use basic computational elements of recommender systems like sparse matrices, matrices operations, database manipulation or random sampling. Also the programming language R and the library ggplot2 [4] was used for data visualisation

Some of the techniques and concepts that we are going to use in this dataset are:
- Knn regression: This is one of the branches of collaborative filtering. We will apply both user and item collaborative filtering. This systems rely on a 2D space with users and items. With neighborhood methods we can determine which users share musical tastes or which concerts are similar and make recommendations based on this similarity.
- Dimensionality reduction: This is another discipline inside collaborative filtering where we transform both items and users to the same latent factor space through non-negative matrix factorisation. For example one of this factors can be how “reggae” or how “punk” a concert can be. We will use the python library “nimfa” [7] for some of this techniques, but we will also implement our own gradient descend method.
- Hybrid recommender systems: Some of the context sensitive recommenders fall into this niche. We will use a contextual pre-filtering technique, where we will select the data first for that specific context (e.g. concerts during the weekends), and then we will apply the standard 2D techniques.
- Context Aware Recommendations: We will include directly the context in our predictions, by applying matrix factorisation techniques that consider more than one dimension. In this extra dimensions we can use things like location, age of the user and so on.
- Measuring performance of recommender systems: How can we know if our predictions are accurate?. How can we measure it?. We will use two approaches. First RMSE will allow us to measure how different is our predicted rating from the real one in a scale from 0 to 5. Second, a topN approach [17] will tell us how accurate our predictions are in terms of a best bet approach. This is used in many commercial systems where the top best N recommendations are shown.
- Data Analysis and visualisation: We will collect statistics about the dataset that will help us to explain some of our results, techniques and the nature of our data. Things like how skewed our ratings are, which countries are the most active, or what is the mean number of concerts that users went to, are some of this useful stats. We will also use data visualisation libraries to better explain this numbers.

