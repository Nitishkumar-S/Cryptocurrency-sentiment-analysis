#Import libraries
import tweepy
from textblob import TextBlob
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Function that cleans tweets
def cleanTwt(twt):
  twt=re.sub("#","",twt)    #removes all hash from hashtags
  twt=re.sub('\\n','',twt)  #removes \n from tweets
  twt=re.sub('https?:\/\/\S+','',twt)   #removes hyperlinks from tweets
  twt=re.sub('@','',twt)  #removes all @ from tweets
  return twt

#Function to get Subjectivity
def getSubjectivity(twt):
  return TextBlob(twt).sentiment.subjectivity
#Function to get Polarity
def getPolarity(twt):
  return TextBlob(twt).sentiment.polarity


#function to get text sentiment
def getSentiment(score):
  if score<0:
    return 'Negative'
  elif score==0:
    return 'Neutral'
  else:
    return 'Positive'


if __name__=="__main__":
    
    login=pd.read_csv("login.csv")     #The api credentials were saved in a file named 'login.csv' which had four keys, api key, api consumer secret, access token key, acccess token secret
                                       #format of login.csv file: column names are key names and entries are the key values
    
    #Get twitter API credentials from login file
    consumerKey=''.join(login['apikey'])
    consumerSecret=''.join(login['apisecretkey'])
    accessToken=''.join(login['accesstoken'])
    accessTokenSecret=''.join(login['accesstokensecret'])

    #Create authentication object
    authenticate= tweepy.OAuthHandler(consumerKey,consumerSecret)

    #Set access token and access token secret
    authenticate.set_access_token(accessToken,accessTokenSecret)

    #Create API object
    api = tweepy.API(authenticate, wait_on_rate_limit=True)

    #Gather 2000 tweets about bitcoin and filter out any retweets
    search_term = "#Dogecoin -filter:retweets"

    #Create a cursor object
    tweets= tweepy.Cursor(api.search, q=search_term, lang='en', since='2021-05-07',tweet_mode='extended').items(2000)

    #Store tweets in variable and get full texts
    all_tweets = [tweet.full_text for tweet in tweets]

    #Create a dataframe with the column name "Tweets" to store tweets
    df = pd.DataFrame(all_tweets, columns=['Tweets'])
    print("Tweets after pulling from twitter")
    #First 5 rows of data
    print(df.head())

    print()
    #to show dimensions
    df.shape

    #Clean tweets
    df['Cleaned_tweets']=df['Tweets'].apply(cleanTwt)
    print("\nAfter cleaning\n")
    #show first 5 rows
    print(df.head(5))

    #create two new columns to store 'subjectivity' an 'polarity' by calling above created functions
    df['Subjectivity']=df['Cleaned_tweets'].apply(getSubjectivity)
    df['Polarity']=df['Cleaned_tweets'].apply(getPolarity)
    print("Subjectivity and polarity added\n")
    #show first 5 rows
    df.head(5)

    #create column to store text sentiment
    df['Sentiment']=df['Polarity'].apply(getSentiment)
    print("Sentiment column added\n")
    #show first 5 rows
    print(df.head())

    # create a scatter plot to show the subjectivity and the polarity
    plt.figure(figsize=(8,6))

    for i in range(df.shape[0]):
        plt.scatter(df["Polarity"][i], df["Subjectivity"][i], color="Purple")

    plt.title("Sentiment Analysis Scatter Plot")
    plt.xlabel('Polarity')
    plt.ylabel('Subjectivity')
    plt.show()

    #create a bar plot to show count of positive, neutral and negative tweets
    df['Sentiment'].value_counts().plot(kind='bar')
    plt.title('Sentiment Analysis Bar Plot')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()
