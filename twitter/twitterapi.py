import requests
import json
import mariadb
from datetime import datetime
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from gensim import models, corpora
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import string
import re
from price import config

def getlist():

    access_token = config.key_twitter

    # Define the base URL for the Twitter API
    base_url = 'https://api.twitter.com/2/'

    # Define the endpoint you want to hit (in this case, we're getting tweets from a list)
    # Replace 'LIST_ID' with the ID of the list you want to get tweets from
    endpoint = 'lists/1665741301575213057/tweets'

    # Define the parameters for your request
    parameters = {
        'tweet.fields': 'public_metrics,created_at',
        'expansions': 'author_id,entities.mentions.username',
        'user.fields': 'created_at,public_metrics',
        'max_results': 100
    }

    # Create your headers
    headers = {
        'Authorization': 'Bearer ' + access_token,
        'Content-Type': 'application/json'
    }

    # Make the request
    response = requests.get(base_url + endpoint, headers=headers, params=parameters)
    # Parse the response
    tweets = json.loads(response.text)
    with open('foo.txt', 'w') as f:
        f.write(response.text)
    return tweets

def gettweets():


    access_token = config.key_twitter

    # Define the base URL for the Twitter API
    base_url = 'https://api.twitter.com/2/'

    # Define the endpoint you want to hit (in this case, we're searching for tweets)
    endpoint = 'tweets/search/recent'

    # Define the parameters for your request (in this case, we're looking for tweets with a specific hashtag)
    parameters = {
        'query': 'Crypto -#loyal -LOYAL -Аirdrop -airdrop -RT -MEV -"mentorship program" -"I claimed" -#Airdrop -"Earn 100% daily Profit"',
        'tweet.fields': 'public_metrics,created_at',
        'expansions': 'author_id,entities.mentions.username',
        'user.fields': 'created_at,public_metrics',
        'max_results': 100
    }

    # Create your headers
    headers = {
        'Authorization': 'Bearer ' + access_token,
        'Content-Type': 'application/json'
    }

    # Make the request
    response = requests.get(base_url + endpoint, headers=headers, params=parameters)
    # Parse the response
    tweets = json.loads(response.text)
    with open('foo.txt', 'w') as f:
        f.write(response.text)
    return tweets

def textmining(tweets):

    # Liste, um alle Wörter in den Tweets zu speichern
    words = []

    for tweet in tweets['data']:
        text = tweet['text']
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])
        tokenized_words = word_tokenize(text)
        words.extend(tokenized_words)

    # Entferne Stopwörter und Nicht-Wörter
    stop_words = set(stopwords.words('english'))
    stop_words.add("https")
    words = [word for word in words if word.isalpha() and word not in stop_words]

    # Perform topic modeling
    dictionary = corpora.Dictionary([words])
    corpus = [dictionary.doc2bow(words)]
    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary)

    # Print the topics and their top words
    topics = lda_model.print_topics(num_topics=5, num_words=5)
    for topic in topics:
        topic

    topics = lda_model.show_topics(num_topics=5, num_words=5, formatted=False)
    topic_data = [{'topic_id': topic[0], 'words': [word[0] for word in topic[1]]} for topic in topics]

    # Zähle die Häufigkeit der Wörter
    word_counts = Counter(words)

    # Drucke die 10 häufigsten Wörter
    print(word_counts.most_common(10))
    return word_counts, topic_data

def counthashtags(tweets):
    hashtags = {}
    sentiments = {"positive": 0, "neutral": 0, "negative": 0}
    analyzer = SentimentIntensityAnalyzer()

    for tweet in tweets['data']:
        text = tweet['text']
        words = text.split()

        # Perform sentiment analysis using VADER
        sentiment_scores = analyzer.polarity_scores(text)
        compound_score = sentiment_scores['compound']

        # Classify the sentiment
        if compound_score >= 0.05:
            sentiments["positive"] += 1
        elif compound_score > -0.05 and compound_score < 0.05:
            sentiments["neutral"] += 1
        else:
            sentiments["negative"] += 1

        for word in words:
            if word.startswith("#"):
                lowercase_hashtag = word.lower()
                if lowercase_hashtag in hashtags:
                    hashtags[lowercase_hashtag] += 1
                else:
                    hashtags[lowercase_hashtag] = 1

    sorted_hashtags = sorted(hashtags.items(), key=lambda x: x[1], reverse=True)
    for hashtag, count in sorted_hashtags:
        print(f"{hashtag}: {count}")

    return hashtags, sentiments

def dbcon():
    # Verbindung zur Datenbank herstellen
    conn = mariadb.connect(user=config.duser, password=config.dpassword, host=config.dhost, database=config.ddatabase)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS hashtags (
            id INT AUTO_INCREMENT PRIMARY KEY,
            fetch_id INT NOT NULL,
            hashtag VARCHAR(255) NOT NULL,
            count INT NOT NULL,
            created_at DATETIME NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sentiments (
            id INT AUTO_INCREMENT PRIMARY KEY,
            fetch_id INT NOT NULL,
            positive INT NOT NULL,
            neutral INT NOT NULL,
            negative INT NOT NULL,
            created_at DATETIME NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS words (
    id INT AUTO_INCREMENT PRIMARY KEY,
    fetch_id INT NOT NULL,
    word VARCHAR(255) NOT NULL,
    count INT NOT NULL,
    created_at DATETIME NOT NULL
    )

    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS fetches (
            id INT AUTO_INCREMENT PRIMARY KEY,
            created_at DATETIME NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS topics (
            id INT AUTO_INCREMENT PRIMARY KEY,
            fetch_id INT NOT NULL,
            topic_data TEXT NOT NULL,
            created_at DATETIME NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tweets (
            id VARCHAR(255) PRIMARY KEY,
            text TEXT NOT NULL,
            retweet_count INT NOT NULL,
            reply_count INT NOT NULL,
            like_count INT NOT NULL,
            quote_count INT NOT NULL,
            impression_count INT NOT NULL,
            fetch_id INT NOT NULL,
            created_at DATETIME NOT NULL,
            author_id BIGINT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS twitter_users (
            id BIGINT PRIMARY KEY,
            name VARCHAR(255),
            username VARCHAR(255),
            created_at DATETIME,
            description TEXT,
            entities JSON,
            location VARCHAR(255),
            pinned_tweet_id BIGINT,
            profile_image_url VARCHAR(255),
            protected BOOLEAN,
            followers_count INT,
            following_count INT,
            tweet_count INT,
            listed_count INT,
            url VARCHAR(255),
            verified BOOLEAN,
            fetch_id INT)
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mentioned_users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            tweet_id VARCHAR(255) NOT NULL,
            author_id BIGINT NOT NULL,
            mentioned_user_id BIGINT NOT NULL,
            mentioned_user_username VARCHAR(500),
            fetch_id INT
        )
    """)
    return cur, conn

def insertfetch(cur, conn):
    current_datetime = datetime.now()
    cur.execute(
        "INSERT INTO fetches (created_at) VALUES (?)",
        (current_datetime,)
    )
    conn.commit()
    cur.execute(f"SELECT MAX(id) FROM fetches")
    last_id = cur.fetchone()[0]
    print(f'Anfrag-Nr. {last_id}')
    return last_id


def insertwords(cur, fetch_id, word_counts):
    current_datetime = datetime.now()
    for word, count in word_counts.most_common(10):
        cur.execute(
            "INSERT INTO words (fetch_id, word, count, created_at) VALUES (?, ?, ?, ?)",
            (fetch_id, word, count, current_datetime)
        )
# Insert sentiment counts into the database
def insertsent(cur, sentiments, fetch_id):
    current_datetime = datetime.now()
    print(sentiments)
    print(current_datetime)
    cur.execute(
        "INSERT INTO sentiments (fetch_id, positive, neutral, negative, created_at) VALUES (?, ?, ?, ?, ?)",
        (fetch_id, sentiments["positive"], sentiments["neutral"], sentiments["negative"], current_datetime)
    )

def inserthashtag(cur, hashtags, fetch_id):
    for hashtag, count in hashtags.items():
        current_datetime = datetime.now()
        cur.execute(
            "INSERT INTO hashtags (fetch_id,hashtag, count, created_at) VALUES (?, ?, ?, ?)",
            (fetch_id, hashtag, count, current_datetime)
        )

def inserttopics(cur, fetch_id, topic_data):
    current_datetime = datetime.now()
    cur.execute(
        "INSERT INTO topics (fetch_id, topic_data, created_at) VALUES (?, ?, ?)",
        (fetch_id, json.dumps(topic_data), current_datetime)
    )

# Datenbankverbindung schließen
def close(cur, conn):
    cur.close()
    conn.commit()
    conn.close()

def extract_urls(tweets):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    for tweet in tweets['data']:
        text = tweet['text']
        urls = re.findall(url_pattern, text)
        print(f'{urls}\n')

def inserttweet(cur, fetch_id, tweet):
    tweet_id = tweet['id']
    text = tweet['text']
    metrics = tweet['public_metrics']
    retweet_count = metrics['retweet_count']
    reply_count = metrics['reply_count']
    like_count = metrics['like_count']
    quote_count = metrics['quote_count']
    author_id = tweet['author_id']
    impression_count = metrics.get('impression_count', 0)  # Not all tweets have this field
    created_at = tweet['created_at']
    b = created_at
    c = b.replace('T', ' ')
    d = c.replace('Z', '')

    cur.execute(
        "INSERT INTO tweets (id, text, retweet_count, reply_count, like_count, quote_count, impression_count, fetch_id, created_at, author_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (tweet_id, text, retweet_count, reply_count, like_count, quote_count, impression_count, fetch_id, d, author_id)
    )
    entities = tweet.get('entities', {})
    if 'mentions' in entities:
        mentions = entities['mentions']

        # Iterate over each mention
        for mention in mentions:
            mentioned_user_id = mention['id']
            mentioned_user_name = mention['username']
            # Insert the mentioned user data into the database
            cur.execute("""
                       INSERT INTO mentioned_users (tweet_id, author_id, mentioned_user_id, mentioned_user_username, fetch_id) 
                       VALUES (?, ?, ?, ?, ?)
                   """, (tweet_id, author_id, mentioned_user_id, mentioned_user_name, fetch_id))
def insert_user(cur, fetch_id, user):
    # Bereite die SQL-Anweisung vor
    sql = """
                INSERT INTO twitter_users (id, name, username, created_at, description, entities, location, pinned_tweet_id, profile_image_url, protected, followers_count, following_count, tweet_count, listed_count, url, verified, fetch_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
    # Bereite die Daten für die SQL-Anweisung vor
    
    created_at = user['created_at']
    b = created_at
    c = b.replace('T', ' ')
    d = c.replace('Z', '')
    values = (
        user['id'],
        user['name'],
        user['username'],
        d,
        user.get('description'),
        json.dumps(user.get('entities', {})),
        user.get('location'),
        user.get('pinned_tweet_id'),
        user.get('profile_image_url'),
        user.get('protected'),
        user['public_metrics']['followers_count'],
        user['public_metrics']['following_count'],
        user['public_metrics']['tweet_count'],
        user['public_metrics']['listed_count'],
        user.get('url'),
        user.get('verified'),
        fetch_id

    )
    cur.execute(sql, values)


def run(fetch_id):


    cur, conn = dbcon()
    tweets = getlist()
    for user in tweets['includes']['users']:
        try:
            insert_user(cur, fetch_id, user)
        except Exception as e:
            print('User-ID bereits vorhanden')
            print(str(e))

    for tweet in tweets['data']:
        try:
            inserttweet(cur, fetch_id, tweet)
        except Exception as e:
            print('Tweet-ID bereits vorhanden')
            print(str(e))
    #extract_urls(tweets)
    word_counts, topic_data = textmining(tweets)
    insertwords(cur,fetch_id, word_counts)
    inserttopics(cur, fetch_id, topic_data)

    hashtags, sentiments = counthashtags(tweets)
    insertsent(cur, sentiments, fetch_id)
    inserthashtag(cur, hashtags, fetch_id)

    close(cur, conn)
    # Print the number of tweets
    print(len(tweets['data']))