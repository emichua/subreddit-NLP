# Project 3: Web APIs & NPL

## Table of Contents:
* Problem Statement
* Executive Summary
* Conclusions and Recommendations

### Problem Statement:
Reddit is a massive collection of forums where people can share news and content or comment on other people’s posts. Reddit is broken up into more than a million communities known as “subreddits,” each of which covers a different topic. The name of a subreddit begins with “r/,” which is part of the URL that Reddit uses. For example, r/nba is a subreddit where people talk about the National Basketball Association. A "post" is where the community share content by stories, links, images, and videos. A "comment" provides discussions on posts. And both comments & posts can be scored by being upvoted or downvoted.

Yet, there is a dilemma, what if we wanted to gather data and model mulitple "subreddits"? This is difficult to compare such information without a classifier.

Thus, can we use supervised machine learning to classify similar content from two different web sources?

How do we investigate this problem?

I scraped about 4000 posts from two chosen subreddits. Each subreddit I scraped was about 2000 posts by using Reddit's API. Then, I used natural language processing to train a classifier model to check which post came for the correct subreddit. The classification models I decided to use were Logistic Regression, Bernoulli Naive Bayes, Bagged Decision Tree, and Random Forest which we evaluated on accuracy scores and results from confusion matrices.

### Executive Summary:
I begin by pulling the data from the two subreddits by using Reddit's API. The subreddits that I pulled were the r/mbti and r/Horoscope subreddits. The data that was imported was in JSON format. Therefore, I decided to create dataframes in Pandas to have easier access to clean and multiplate through the data.

Once, I looked through the dataframes, I looked for particular subfields using the Reddit's API data dictionary. I focused on the title, created_utc, author, selftext, and subreddit features. I chose these as the subfields because I wanted the best features for our modeling.

Next, I did some data cleaning. I checked for duplicate posts and missing values in each of the dataframes. Lastly, I combined into two subreddit into one dataframe, named 'subreddit'.

Then, I did some exploratory data analysis. I first showed the date range for the subreddit I have scraped. Thinking back on my problem statement, I want to detemine similar content in both of the datasets, thus, I do this by looking at the frequently occurring words in each dataframe. I did this by using an NLP functions called stemming and countvectorizer.I chose to display bar graphs that had the top 10 frequently occurring single gram word & bigram words in each subreddit.

Next, I preprocess my data. I dropped the author and selftext feature because I do not need it for modeling. I mapped our target variable: subreddit into a binary classification. I did some more NLP processing. I used lemmatization, stemming, and stopwords to analyize my dataset futher. Then, I created our X feature and target variable and did a train-test split. I decided to change our X feature as a lemmatised version for our modeling. Lastly, I determined the basline score to compare to our models' results.

Finally, I modeled four different classification models. I modeled Logistic Regression, Bernoulli Naive Bayes, Bagged Decision Tree, and Random Forest. I also created a confusion matrix for each model to have further insights on each of my models. I wanted to see how well our models were able to correctly classify where each post came from. In the end, I focused on accuracy score and the bias-variance tradeoff from each model to determined which model was the best to answer my problem statement.

### Conclusion and Recommendations

| | Model | Training Accuracy Score  | Testing Accuracy Score | Correctly Classified |Misclassified| AUC|
|---:|:-------------|:-----------|:------|:------|:----------|:---------|
| 1 | Logistic Regression | 0.963  | 0.930  | 945  | 71|0.7902|
| 2 | Bernoulli Naive Bayes | 0.974  | 0.948 | 971 | 54|0.8437|
| 3 | Bagged Decision Tree | 0.991|0.895|918|107|0.9799|
| 4 | Random forest |0.998|0.913|936|89|0.9822|



All the classification models: Logistic Regression, Bernoulli Naive Bayes, Bagged Decision Tree, and Random Forest surpassed the baseline accuracy. 

**Bernoulli Naive Bayes Classification Model** was the best model to test our training data because it was able to manage well with unknown data according to the testing accuracy score. Despite having lower AUC score than both bagged decision tree and random forest.

However, the model was still overfit because of low bias and high variance.

Despite the overfitting, this model was able to classified similar content from two different web sources: r/mbti and r/astrology. Also it was able to identify where each post came from which subreddit. It had one of the highest in correctly classifying posts which was 971. Therefore, Reddit will be able to implement this model for their studies on this data science topic.

Yet, this model still had its limitations. Some of the posts had similar titles and incorrect spelling. It was not able to identify these mishaps because of NLP transformer. In other words, both of these mishaps could of swayed our model results.

Also, our model can improve it's accuracy if we further tuned our hyperparameters. Additionally, we could of instantiate the PolynomialFeatures before our model to decrease the overfitting.

So we still have some recommendations:

- increase the stopword list with more nouns to have a better predictable model (i.e. ‘like’)
- Each of the subreddits change over time, thus the result might change.
- a different model that we had not yet modeled (i.e. SLM, Adaboost)



