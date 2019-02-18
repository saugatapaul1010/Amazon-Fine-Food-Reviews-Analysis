# Amazon Fine Food Reviews Analysis.

This repository contains a series of experiments performed on the Amazon Reviews dataset. This includes data visualization techniques like PCA and T-SNE. It also includes curating, de-duping and cleaning the dataset to perform a series of NLP related tasks. You will learn about a series of models which includes KNN, Naive Bayes, Logistic Regression, SVMs, Decision Trees, GBDTs, XGBoosts, Random Forests etc. You weill also learn about applying clustering techniques like KMeans, Agglomerative clustering and DBSCAN. You will further learn about Matrix Factorization usng Truncated SVD followed by KMeans clustering.

The dataset can be downloaded from : https://www.kaggle.com/snap/amazon-fine-food-reviews

#### Basic information about the downloaded dataset<br>

Number of reviews: 568,454<br>
Number of users: 256,059<br>
Number of products: 74,258<br>
Timespan: Oct 1999 - Oct 2012<br>
Number of Attributes/Columns in data: 10 

#### Attribute Information:

1. Id
2. ProductId - unique identifier for the product
3. UserId - unqiue identifier for the user
4. ProfileName
5. HelpfulnessNumerator - number of users who found the review helpful
6. HelpfulnessDenominator - number of users who indicated whether they found the review helpful or not
7. Score - rating between 1 and 5
8. Time - timestamp for the review
9. Summary - brief summary of the review
10. Text - text of the review


#### Objective:
Our main objective for this analysis is to train a model which can seperate the postive and negative reviews. By looking at the Score column we can make out that the review is positive or not. But we don't need to implement any ML here. A simple if-else condition will make us do this. So for this problem, we will put our focus on to the Review text. The text is the most important feature here if you may ask. Based on the review text we will build a prediction model and determine if a future review is positive or negative.

#### While pre-processing the original dataset we have taken into consideration the following points.

1. We will classify a review to be positive if and only if the corresponding Score for the given review is 4 or 5.
2. We will classify a review to be negative if and only if the corresponding Score for the given review is 1 or 2.
3. We will ignore the reviews for the time being which has a Score rating of 3. Because 3 can be thought of as a neutral review. It's neither negative nor positive.
4. We will remove the duplicate entries from the dataset. 
5. We will train our final mdel using four featurizations -> bag of words model, tf-idf model, average word-to-vec model and tf-idf weighted word-to-vec model.
6. So at end of the training the model will be trained on the above four featurizations to determine if a given review is positive or negative (Determining the sentiment polarity of the Amazon reviews)

#### Data Cleaning Stage:

This Ipython notebook contains all the steps needed to clean and process our data. The steps include:
1. Loading the oiginal dataset.

2. Perform Exploratory Data analysis on the Amazon Fine Food Reviews Dataset.

3. Perform Data Cleaning Stage 1: Remove Duplicate Data points from the original dataset and arrange the reviews in descending order of time such that the oldest reviews appear at the top and the newest reviews appear at the bottom.

4. Perform Data Cleaning Stage 2: Pre-processing of the review texts.

(a) Remove urls from text using python. Visit: https://stackoverflow.com/a/40823105/4084039

(b) Remove HTML tags from all the reviews.

(c) Exapnd the most common English contractions present in the reviews. Visit https://en.wikipedia.org/wiki/Wikipedia:List_of_English_contractions for a list of all contractions available in English Language. Visit ttps://stackoverflow.com/a/47091490/4084039 for more information on how to expand contractions.

(d) Remove words with numbers using python. Visit https://stackoverflow.com/a/18082370/4084039 for ref.

(e) Remove Punctuations from all the reviews using Regular Expressions. This will keep only words containing letters A-Z and a-z. This will remove all punctuations, special characters etc. https://stackoverflow.com/a/5843547/4084039 for more reference.

(f) Remove words like 'zzzzzzzzzzzzzzzzzzzzzzz', 'testtting', 'grrrrrrreeeettttt' etc. Preserves words like 'looks', 'goods', 'soon' etc. We will remove all such words which has three consecutive repeating characters. Visit #https://stackoverflow.com/questions/37012948/regex-to-match-an-entire-word-that-contains-repeated-character

(g) Use SnowballStemmer or PortalStemmer to remove all the frequently occuring stopwords from each review. Having said that, we will preserve stopwords like 'not','nor','no'. Because, if you think intuitively, words like 'not' might be used more often while dealing with negative reviews. So we will keep them!

(h) We will remove all the stemmed words which has a length greater than 15. By looking at the distribution of the length of words we can see that most stemmed words present in the reviews has lengths between 4 and 10. Words which has length greater than 15 are very very very few as compared to other words. So we will discard these words from the reviews when we process them. It means we will consider only those words whose length is greater than 2 and less than 16.

(i) After cleaning the dataset we will save the processed dataset in a DB called totally_processed_DB.sqlite. We will use this DB for all our future works.

5. We will now plot a Histogram to check the distribution of positive as well as negative reviews in the pre-processed dataset.

6. Featurizations: After processing the data we will use the following 5 featurization techniques to build ML/DL models on top of them.

(a) Bag of Words featurization.

A bag-of-words model, or BoW for short, is a way of extracting features from text for use in modeling, such as with machine learning algorithms.
The approach is very simple and flexible, and can be used in a myriad of ways for extracting features from documents. Suppose we have N reviews in our dataset and we want to convert the words in our reviews to vectors. We can use BOW as a method to do this. What it does is that for each unique word in the data corpus, it creates a dimension. Then it counts how many number of times a word is present in a review. And then this number is placed under that word for a corresponding review. We will get a Sparse Matrix representation for all the worods inthe review.

Let's look at this example of 2 reviews below :

r1 = {"The food is great, ambience is great"} and  </br>r2 = {"I love this food"}

At first the words will be extracted from r1 and r2.

r1' = {"The", "food", "is", "great", "ambience", "is", "great"} and r2' = {"I", "love", "this", "food"}

Now using r1' and r2' we will create a vector of unique words -> V =  {"The", "food", "is", "great", "ambience", "I", "love", "this"}

Now here's how the vector representation will look like for each reviews r1 and r2, when we make use of the vector 'V' created above.

r1_vector = [1,1,2,2,1,0,0,0] and r2_vector = [0,1,0,0,0,1,1,1]

In r1 since, "great" and "is" occurs twice, we have set the count to 2. If a words doesn't occur in a review we will set the count to 0. Although "is" a stopword, the example above is intended to make you understand how bag of words work.

(b) Bi-Grams and n-Grams: Here instead of taking just one word we will use two or n consecutive words to build featurizations. This helps us in retaining the sequence information when a word has a tendency to occur with another word(s)

(c) TF-IDF featurization.

In information retrieval, tf–idf or TFIDF, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.

Let's assume we have data corpus D, which contains N reviews {r1,r2,r3,r4...rN}. Let's say our review r1 contains the following words {w1,w2,w3,w1,w9,w6,w7,w9,w9}.

TF or Term Frequency for a word is basically the number of times a word occurs in a review divided by the total number of words present in that same review.
For example, in the text corpus that we have considered in the above example, the TF for word w1 is (2/9) and for word w9 is (1/3). Intuitively, higher the occurence of a word in a text is, greater will be its TF value. TF values lies between 0 and 1.

IDF or Inverse Document Frequency for a word is given by the formula log(N/n), where 'N' is equal to the total number of reviews in the corpus 'D' and 'n' refers to the number of reviews in 'D' which contains that specific word. Intuitively, IDF will be higher for words which occur rarely and will be less for words which occurs more frequently. IDF values are more than 0.

So for each word in each review we will consider the product of (TF x IDF), and represent it in a d dimensional vector. 

TF-IDF basically doesn't consider the semantic meaning of words. But what is does is that it gives more importance to words which occurs less frequently in the whole data corpus and also gives much importance to the most frequent words that occurs in each review.

(d) Average Word2Vec models.

In this model we convert each word present in a review to vectors. For each sentence we will compute the average word to vec representation. Let's look at the below demo example.

Suppose we have N words in a sentence {w1,w2,w3,w4,w5,w6 ... , wN}. We will convert each word to a vector, sum them up and divide by the total number of words (N) present in that particular sentence. So our final vector will look like (1/N) * [word2vec(w1) + word2vec(w2) + word2vec(w3) .... + word2vec(wN)]

(e) TFIDF weighted Average Word2Vec model.

In this model we convert each word present in a review to vectors. For each sentence we will compute the tf-idf average word to vec representation. Let's look at the below demo example.

Suppose we have N words in a sentence {w1,w2,w3,w4,w5,w6 ... , wN}. We will compute the tf-idf for each word in a review for all reviews. Lets say the corresponding tf-idfs are {t1,t2,t3,t4,t5,t6......tN}. We will convert each word to a vector, sum them up and divide by the summation of tf-idf vectors for all words present in that particular sentence. So our final vector will look like [1/(t1+t2+t3+t4+t5+t6+ ..... +tN)] * [word2vec(w1) + word2vec(w2) + word2vec(w3) .... + word2vec(wN)]

7. After performing the featurization tasks, we will se 100K reviews as our train data. We will use 40K datapoints to caliberate our model and 30K data points to predict the performance of the model on new unseen data.

