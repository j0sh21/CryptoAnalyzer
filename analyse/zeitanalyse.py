import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import mariadb
import json
from wordcloud import WordCloud
from collections import defaultdict
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
import nltk
from nltk.corpus import stopwords
import tkinter as tk
from tkcalendar import Calendar
from datetime import datetime, time

def dbconnect():
    # Verbindung zur Datenbank herstellen
    conn = mariadb.connect(user='j0n1', password='12345', host="192.168.178.89", database="test")
    cur = conn.cursor()
    return cur, conn

def plothash(cur,start, end):
    # Daten aus der Datenbank abrufen
    cur.execute(f"SELECT created_at, count, hashtag FROM hashtags where created_at between '{start}' and '{end}' GROUP BY created_at, hashtag order by count desc")
    data = cur.fetchall()

    # Daten in einen pandas DataFrame umwandeln
    df = pd.DataFrame(data, columns=['created_at', 'count', 'hashtag'])

    # Das Datum als Index setzen
    df.set_index('created_at', inplace=True)

    # Die Hashtags nach ihrer Gesamtzahl sortieren
    hashtag_counts = df.groupby('hashtag')['count'].sum().sort_values(ascending=False)

    # Nur die Top 20 Hashtags auswählen
    top_hashtags = hashtag_counts.head(20).index

    # Für jeden der Top 20 Hashtags einen Plot erstellen
    for hashtag in top_hashtags:
        hashtag_df = df[df['hashtag'] == hashtag]['count']
        hashtag_df = hashtag_df.resample('5T').sum().fillna(
            0)  # Normiere die Daten auf eine tägliche Zeitreihe und setze fehlende Werte auf 0
        hashtag_df = hashtag_df.rolling(window=12).mean()
        hashtag_df = (hashtag_df - hashtag_df.min()) / (hashtag_df.max() - hashtag_df.min())
        hashtag_df.plot(label=hashtag, marker='x')
    plt.legend()
    plt.show()

def networkanalyse(cur, start, end):
    G = nx.Graph()
    cur.execute(f"SELECT text FROM tweets where created_at between '{start}' and '{end}'")
    tweets = cur.fetchall()

    for tweet in tweets:
        text = tweet[0]  # extract the text from the tuple
        words = text.split()
        hashtags_in_tweet = [word.lower() for word in words if word.startswith("#")]
        for i in range(len(hashtags_in_tweet)):
            for j in range(i+1, len(hashtags_in_tweet)):
                if G.has_edge(hashtags_in_tweet[i], hashtags_in_tweet[j]):
                    # increase weight by 1
                    G[hashtags_in_tweet[i]][hashtags_in_tweet[j]]['weight'] += 1
                else:
                    # new edge. add with weight=1
                    G.add_edge(hashtags_in_tweet[i], hashtags_in_tweet[j], weight=1)
    plt.figure(figsize=(10,10))
    pos = nx.spring_layout(G, k=0.15)
    nx.draw_networkx(G, pos, node_size=20, node_color='blue', font_size=10, edge_color='grey')
    plt.show()
    return G


def networkanalyse_follower(cur, start, end):
    G = nx.Graph()
    cur.execute(f"""
    SELECT tweets.text, tweets.author_id 
    FROM tweets
    INNER JOIN twitter_users ON tweets.author_id = twitter_users.id 
    WHERE tweets.author_id IS NOT NULL AND twitter_users.followers_count > 500 and tweets.created_at between '{start}' and '{end}'
    """)
    tweets = cur.fetchall()

    for tweet in tweets:
        text, author_id = tweet
        cur.execute("SELECT followers_count FROM twitter_users WHERE id = %s", (author_id,))
        follower_count = cur.fetchone()[0]

        words = text.split()
        hashtags_in_tweet = [word.lower() for word in words if word.startswith("#")]
        for i in range(len(hashtags_in_tweet)):
            for j in range(i + 1, len(hashtags_in_tweet)):
                if G.has_edge(hashtags_in_tweet[i], hashtags_in_tweet[j]):
                    G[hashtags_in_tweet[i]][hashtags_in_tweet[j]]['weight'] += follower_count
                else:
                    G.add_edge(hashtags_in_tweet[i], hashtags_in_tweet[j], weight=follower_count)

    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, k=2.15)

    # Calculate the node sizes based on weight. Note that this is an arbitrary calculation and you might want to adjust it to better suit your needs.
    node_sizes = [np.sqrt(np.abs(G.degree(node, weight='weight')) * 100) for node in G.nodes()]

    nx.draw_networkx(G, pos, node_size=node_sizes, node_color='blue', font_size=10, edge_color='grey')
    plt.show()
    return G


def mostengagingtopics(cur, start, end):
    # First, fetch engagement data and join it with topic data
    cur.execute(
        f"SELECT topics.topic_data, tweets.retweet_count, tweets.reply_count, tweets.like_count, tweets.quote_count FROM topics JOIN tweets ON topics.fetch_id = tweets.fetch_id where tweets.created_at between '{start}' and '{end}'")
    data = cur.fetchall()
    df = pd.DataFrame(data, columns=['topic_data', 'retweets', 'replies', 'likes', 'quotes'])

    # Convert topic_data from JSON to list of words, and then to a string of words
    df['topics'] = df['topic_data'].apply(lambda x: [' '.join(topic['words']) for topic in json.loads(x)])

    # Now we "explode" the topics column, so we have one row per topic
    df = df.explode('topics')

    # Calculate engagement for each topic as the sum of retweets, replies, likes, and quotes
    df['engagement'] = df['retweets'] + df['replies'] + df['likes'] + df['quotes']

    # Group by topic and sum the engagement
    engagement_by_topic = df.groupby('topics')['engagement'].sum().sort_values(ascending=False)

    # Plot the top 10 most engaging topics
    engagement_by_topic.head(10).plot(kind='bar', figsize=(12, 6))
    plt.title("Most Engaging Topics")
    plt.ylabel("Engagement")

    # Rotate the x-axis labels to improve readability
    plt.xticks(rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.28)
    plt.show()


def sentimentovertime(cur, start, end):
    cur.execute(f"SELECT created_at, positive, neutral, negative FROM sentiments where created_at between '{start}' and '{end}' ORDER BY created_at")
    data = cur.fetchall()
    df = pd.DataFrame(data, columns=['created_at', 'positive', 'neutral', 'negative'])
    df.set_index('created_at', inplace=True)
    df.plot()
    plt.show()

def trendingtopics(cur, start, end):
    cur.execute(f"SELECT created_at, count, hashtag FROM hashtags where count > 2 and created_at between '{start}' and '{end}' ORDER BY created_at")
    data = cur.fetchall()
    df = pd.DataFrame(data, columns=['created_at', 'count', 'hashtag'])
    df.set_index('created_at', inplace=True)
    df.groupby('hashtag')['count'].plot()

    df.plot(kind='bar', figsize=(12, 6))
    plt.title("Top Hashtags by Count")
    plt.ylabel("Count")
    plt.show()

def trendingtopicste(cur, start, end):
    # Fetch data from the database
    cur.execute(f"SELECT created_at, topic_data FROM topics where created_at between '{start}' and '{end}' ORDER BY created_at")
    data = cur.fetchall()

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data, columns=['created_at', 'topic_data'])

    # Convert the JSON strings in the 'topic_data' column into Python lists of dictionaries
    df['topic_data'] = df['topic_data'].apply(json.loads)

    # Expand the DataFrame so that each row contains one dictionary from the 'topic_data' list
    df = df.explode('topic_data')

    # Convert the dictionaries in the 'topic_data' column into strings (of sorted words to avoid duplicates)
    df['topic_data'] = df['topic_data'].apply(lambda d: ', '.join(sorted(d['words'])))

    # Count the number of occurrences of each topic
    topic_counts = df.groupby('topic_data')['created_at'].count()

    # Find the topics with the most occurrences (i.e., the trending topics)
    trending_topics = topic_counts.sort_values(ascending=False).head(20)

    # Plot the trending topics
    trending_topics.plot(kind='bar', figsize=(12, 6))
    plt.title("Trending Topics")
    plt.ylabel("Count")

    # Rotate the x-axis labels to improve readability
    plt.xticks(rotation=35, ha='right')
    plt.subplots_adjust(bottom=0.25)
    plt.show()

def toptopics(cur, start,end):
    # Fetch data from the database
    cur.execute(f"SELECT topic_data, positive, neutral, negative FROM sentiments JOIN topics ON sentiments.fetch_id = topics.fetch_id where topics.created_at between '{start}' and '{end}'")
    data = cur.fetchall()

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data, columns=['topic_data', 'positive', 'neutral', 'negative'])

    # Convert topic_data from JSON to list of words, and then to a string of words
    df['topics'] = df['topic_data'].apply(lambda x: [' '.join(topic['words']) for topic in json.loads(x)])

    # Now we "explode" the topics column, so we have one row per topic
    df = df.explode('topics')

    # Calculate the total sentiment score for each topic
    df['total_sentiment'] = df['positive'] - df['negative']

    # Find the topics with the highest total sentiment score
    top_topics = df.groupby('topics')['total_sentiment'].sum().sort_values(ascending=False).head(20)

    # Plot the top topics
    top_topics.plot(kind='bar', figsize=(12, 6))
    plt.title("Top Topics by Sentiment Score")
    plt.ylabel("Total Sentiment Score")

    # Rotate the x-axis labels to improve readability
    plt.xticks(rotation=35, ha='right')
    plt.subplots_adjust(bottom=0.25)

    plt.show()


def topword(cur, start, end):
    cur.execute(f"SELECT word, count FROM words where word <> 'crypto' and created_at between '{start}' and '{end}'")
    data = cur.fetchall()

    # Daten in einen pandas DataFrame umwandeln
    df = pd.DataFrame(data, columns=['word', 'count'])

    # Die Wörter nach ihrer Gesamtzahl sortieren
    word_counts = df.groupby('word')['count'].sum().sort_values(ascending=False)

    # Nur die Top 20 Wörter auswählen
    top_words = word_counts.head(20)

    # Plot erstellen
    top_words.plot(kind='bar', figsize=(12, 6))
    plt.title("Top Words by Count")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha='right')
    plt.show()

def sentiment(cur, start, end):
    cur.execute(
        f"SELECT t.hashtag, s.positive, s.neutral, s.negative FROM hashtags as t JOIN sentiments as s ON t.fetch_id = s.fetch_id where t.hashtag <> '#crypto' and s.created_at between '{start}' and '{end}' order by t.created_at desc")
    data = cur.fetchall()
    df = pd.DataFrame(data, columns=['hashtag', 'positive', 'neutral', 'negative'])

    # Set 'hashtag' as the index
    df.set_index('hashtag', inplace=True)

    df.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title("Topic Sentiment Analysis")
    plt.ylabel("Count")
    plt.xlabel("hashtag")

    # Rotate the x-axis labels to improve readability
    plt.xticks(rotation=45, ha='right')

    plt.show()


def word_cloud_old(cur, start, end):
    cur.execute(f"SELECT word FROM words where created_at between '{start}' and '{end}'")
    data = cur.fetchall()
    text = ' '.join([word[0] for word in data])

    wordcloud = WordCloud(width = 800, height = 800,
                    background_color ='white',
                    stopwords = None,
                    min_font_size = 10).generate(text)

    plt.figure(figsize = (6, 6), facecolor = None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad = 0)

    plt.show()

def pnwords(cur, start, end):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from collections import Counter
    import nltk

    nltk.download('vader_lexicon')

    sia = SentimentIntensityAnalyzer()
    cur.execute(f"SELECT text FROM tweets where created_at between '{start}' and '{end}'")
    data = cur.fetchall()

    # Assuming 'text' column contains the tweet text
    all_words = ' '.join([word[0] for word in data]).split()
    sentiment_scores = {word: sia.polarity_scores(word)['compound'] for word in all_words}

    positive_words = [word for word, score in sentiment_scores.items() if score > 0]
    negative_words = [word for word, score in sentiment_scores.items() if score < 0]

    print('Positive words:', Counter(positive_words).most_common(10))
    print('Negative words:', Counter(negative_words).most_common(10))

def test(cur, start, end):
    G = nx.Graph()
    hashtag_counts = defaultdict(int)
    hashtag_pairs = defaultdict(int)
    cur.execute(f"SELECT text FROM tweets where created_at between '{start}' and '{end}'")
    data = cur.fetchall()
    for row in data:
        tweet_text = row[0]
        hashtags = [word.lower() for word in tweet_text.split() if word.startswith('#')]
        for hashtag in hashtags:
            hashtag_counts[hashtag] += 1
        for i in range(len(hashtags)):
            for j in range(i + 1, len(hashtags)):
                if hashtag_counts[hashtags[i]] >= 5 and hashtag_counts[hashtags[j]] >= 5:
                    if G.has_edge(hashtags[i], hashtags[j]):
                        G[hashtags[i]][hashtags[j]]['weight'] += 1
                    else:
                        G.add_edge(hashtags[i], hashtags[j], weight=1)

    most_common_pairs = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)

    print('Most common co-occurrences of hashtags:', most_common_pairs[:10])
    pos = nx.spring_layout(G, k=2.25)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    top_hashtags = sorted(hashtag_counts, key=hashtag_counts.get, reverse=True)[:5]
    top_6_20 = sorted(hashtag_counts, key=hashtag_counts.get, reverse=True)[6:25]

    node_colors = ['green' if node in top_hashtags else 'yellow' if node in top_6_20 else 'blue' for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            color=node_colors,
            size=[hashtag_counts[node] for node in G.nodes()],
            sizemode='area',
            sizeref=2.*max(hashtag_counts.values())/(40.**2),
            sizemin=4,
            line_width=2
        ),
        text=[node for node in G.nodes()])

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='<br>Network graph made with Python',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    fig.show()



def cchashtags(cur, start, end):


    G = nx.Graph()
    hashtag_counts = defaultdict(int)
    hashtag_pairs = defaultdict(int)
    cur.execute(f"SELECT text FROM tweets where created_at between '{start}' and '{end}'")
    data = cur.fetchall()
    for row in data:
        tweet_text = row[0]
        hashtags = [word.lower() for word in tweet_text.split() if word.startswith('#')]
        for hashtag in hashtags:
            hashtag_counts[hashtag] += 1
        for i in range(len(hashtags)):
            for j in range(i + 1, len(hashtags)):
                if hashtag_counts[hashtags[i]] >= 20 and hashtag_counts[hashtags[j]] >= 20:
                    if G.has_edge(hashtags[i], hashtags[j]):
                        G[hashtags[i]][hashtags[j]]['weight'] += 1
                    else:
                        G.add_edge(hashtags[i], hashtags[j], weight=1)

    most_common_pairs = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)

    print('Most common co-occurrences of hashtags:', most_common_pairs[:10])

    # Draw graph
    pos = nx.spring_layout(G, k=2.25)   # adjust the multiplier as needed
    # Get top 5 hashtags based on frequency
    top_hashtags = sorted(hashtag_counts, key=hashtag_counts.get, reverse=True)[:5]
    top_6_20 = sorted(hashtag_counts, key=hashtag_counts.get, reverse=True)[6:25]
    top_500_hashtags = sorted(hashtag_counts, key=hashtag_counts.get, reverse=True)[:100]
    # Assign colors to nodes
    node_colors = ['green' if node in top_hashtags else 'yellow' if node in top_6_20 else 'blue' for node in G.nodes()]

    # Draw graph

    node_sizes = [hashtag_counts[node] * 1 for node in G.nodes()]  # adjust the multiplier as needed
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors)
    top_500_labels = {node: node for node in G.nodes() if node in top_500_hashtags}
    nx.draw_networkx_labels(G, pos, labels=top_500_labels)
    nx.draw_networkx_edges(G, pos, edgelist=[edge for edge in G.edges(data=True)],
                           width=[v * 0.01 for v in nx.get_edge_attributes(G, 'weight').values()])

    plt.show()


def word_cloud(cur, start, end):
    cur.execute(f"SELECT text FROM tweets where created_at between '{start}' and '{end}'")
    data = cur.fetchall()
    text = ' '.join([word[0] for word in data])
    text_filtered = ''
    for word in text.split():
        if word.startswith(""):
            text_filtered += word + ' '

    stop_words_english = set(stopwords.words('english'))
    stop_words_german = set(stopwords.words('german'))
    stop_words = stop_words_english.union(stop_words_german)
    stop_words.update(['https', 'co', 'rt'])

    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stop_words,
                          min_font_size=10).generate(text_filtered.lower())

    plt.figure(figsize=(6, 6), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)

    plt.show()
def topic_sentiment_ratio(cur, start, end):
    cur.execute(f"SELECT topic_data, positive, negative FROM sentiments JOIN topics ON sentiments.fetch_id = topics.fetch_id where sentiments.created_at between '{start}' and '{end}'")
    data = cur.fetchall()

    df = pd.DataFrame.merge

def word_frequency_over_time(cur, word, start, end):
    # Fetch data from the database
    cur.execute(f"SELECT created_at, count FROM words WHERE word = '{word}' and created_at between '{start}' and '{end}' ORDER BY created_at")
    data = cur.fetchall()

    # Convert the data into a pandas DataFrame
    df = pd.DataFrame(data, columns=['created_at', 'count'])

    # Set 'created_at' as the index
    df.set_index('created_at', inplace=True)

    # Plot the word frequency over time
    df.plot()
    plt.title(f"Frequency of '{word}' Over Time")
    plt.ylabel("Count")
    plt.show()

def analyze_twitter_users(cur):
    # Fetch data from twitter_users table
    cur.execute("SELECT id, name, username, followers_count, following_count, tweet_count, location, verified FROM twitter_users where fetch_id > 1834")
    data = cur.fetchall()

    # Create a DataFrame from the data
    df = pd.DataFrame(data, columns=['id', 'name', 'username', 'followers_count', 'following_count', 'tweet_count', 'location', 'verified'])

    # User Influence: Display top 10 users by follower count
    print("Top 10 users by follower count:")
    print(df.sort_values(by='followers_count', ascending=False).head(10))

    # User Activity: Display top 10 users by tweet count
    print("Top 10 users by tweet count:")
    print(df.sort_values(by='tweet_count', ascending=False).head(10))

    # User Location: Display counts by location
    print("User count by location:")
    print(df['location'].value_counts())

    # User Engagement: Need to join with tweets table
    cur.execute("SELECT twitter_users.id, twitter_users.username, tweets.like_count, tweets.retweet_count, tweets.quote_count FROM twitter_users INNER JOIN tweets ON twitter_users.id = tweets.author_id")
    engagement_data = cur.fetchall()
    engagement_df = pd.DataFrame(engagement_data, columns=['id', 'username', 'like_count', 'retweet_count', 'quote_count'])
    # Aggregate like_count, retweet_count and quote_count for each user
    engagement_df = engagement_df.groupby(['id', 'username']).sum().reset_index()
    # Display top 10 users by total engagement (likes + retweets + quotes)
    engagement_df['total_engagement'] = engagement_df['like_count'] + engagement_df['retweet_count'] + engagement_df['quote_count']
    print("Top 10 users by total engagement:")
    print(engagement_df.sort_values(by='total_engagement', ascending=False).head(10))

    # Tweet to Follower Ratio
    df['tweet_follower_ratio'] = df['tweet_count'] / df['followers_count']
    # Display top 10 users with highest tweet to follower ratio
    print("Top 10 users by tweet to follower ratio:")
    print(df.sort_values(by='tweet_follower_ratio', ascending=False).head(10))

def analyze_sentiment(tweets):
    sentiments = {"positive": 0, "neutral": 0, "negative": 0}
    analyzer = SentimentIntensityAnalyzer()

    for tweet in tweets:
        text = tweet
        sentiment_scores = analyzer.polarity_scores(text)
        compound_score = sentiment_scores['compound']

        # Classify the sentiment
        if compound_score >= 0.05:
            sentiments["positive"] += 1
        elif compound_score > -0.05 and compound_score < 0.05:
            sentiments["neutral"] += 1
        else:
            sentiments["negative"] += 1

    return sentiments

def analyze_all_users(cur):
    # Fetch all user ids
    cur.execute("SELECT id FROM twitter_users where fetch_id > 1834")
    user_ids = [item[0] for item in cur.fetchall()]

    # Loop through all users and analyze sentiments
    for user_id in user_ids:
        analyze_user_sentiment(cur, user_id)

def analyze_user_sentiment(cur, user_id):
    # Get count of tweets for the user
    cur.execute("SELECT COUNT(*) FROM tweets WHERE author_id = %s", (user_id,))
    tweet_count = cur.fetchone()[0]

    # Proceed only if user has more than 10 tweets
    if tweet_count > 10:
        # Fetch tweets of the user
        cur.execute("SELECT text FROM tweets WHERE author_id = %s", (user_id,))
        tweets = cur.fetchall()
        tweets = [item[0] for item in tweets]  # Convert to list of tweets

        sentiments = {"positive": 0, "neutral": 0, "negative": 0}
        analyzer = SentimentIntensityAnalyzer()

        for text in tweets:
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
        print(f"Sentiment analysis for user {user_id}: {sentiments}")
        return sentiments

def user_popularity(cur, start, end):
    cur.execute(f"""
            SELECT twitter_users.id, twitter_users.name, AVG(tweets.like_count) AS avg_likes, AVG(tweets.retweet_count) AS avg_retweets,
            AVG(tweets.reply_count) AS avg_replies, AVG(tweets.impression_count) AS avg_impressions
            FROM twitter_users
            JOIN tweets ON twitter_users.id = tweets.author_id
            WHERE twitter_users.followers_count > 50
            AND tweets.created_at between '{start}' and '{end}' 
            GROUP BY twitter_users.id
            HAVING AVG(tweets.like_count) > 5 OR AVG(tweets.retweet_count) > 5 OR AVG(tweets.reply_count) > 5 OR AVG(tweets.impression_count) > 5
        """)
    data = pd.DataFrame(cur.fetchall(),
                        columns=['id', 'name', 'avg_likes', 'avg_retweets', 'avg_replies', 'avg_impressions'])
    data[['avg_likes', 'avg_retweets', 'avg_replies', 'avg_impressions']] = data[
        ['avg_likes', 'avg_retweets', 'avg_replies', 'avg_impressions']].apply(pd.to_numeric, errors='coerce')

    # sanitizing names
    data['name'] = data['name'].apply(lambda x: ''.join(e for e in x if e.isalnum()))

    plt.figure(figsize=(10, 5))
    data.set_index('name')[['avg_likes', 'avg_retweets', 'avg_replies', 'avg_impressions']].plot(kind='bar')
    plt.title('User Popularity and Engagement')
    plt.show()


def select_date_time_range():
    selected_date_start = None
    selected_date_end = None
    selected_time_start = None
    selected_time_end = None
    datetime_range = {'start': None, 'end': None}  # Container to store date-time values

    def get_selected_range():
        nonlocal selected_date_start, selected_date_end, selected_time_start, selected_time_end
        start_datetime = datetime.combine(selected_date_start, selected_time_start)
        end_datetime = datetime.combine(selected_date_end, selected_time_end)
        datetime_range['start'] = start_datetime.strftime('%Y-%m-%d %H:%M:%S')  # store values in the container
        datetime_range['end'] = end_datetime.strftime('%Y-%m-%d %H:%M:%S')
        window.destroy()  # Close the window here

    def get_selected_date_start(event):
        nonlocal selected_date_start
        selected_date_start = cal_start.selection_get()

    def get_selected_date_end(event):
        nonlocal selected_date_end
        selected_date_end = cal_end.selection_get()

    def get_selected_time_start(event):
        nonlocal selected_time_start
        time_str = time_start.get()
        hours, minutes = map(int, time_str.split(':'))
        selected_time_start = time(hours, minutes)

    def get_selected_time_end(event):
        nonlocal selected_time_end
        time_str = time_end.get()
        hours, minutes = map(int, time_str.split(':'))
        selected_time_end = time(hours, minutes)

    window = tk.Tk()
    window.title("Select Date and Time Range")

    cal_start = Calendar(window, selectmode="day", date_pattern="yyyy-mm-dd")
    cal_start.pack()
    cal_start.bind("<<CalendarSelected>>", get_selected_date_start)

    cal_end = Calendar(window, selectmode="day", date_pattern="yyyy-mm-dd")
    cal_end.pack()
    cal_end.bind("<<CalendarSelected>>", get_selected_date_end)

    time_start_label = tk.Label(window, text="Start Time:")
    time_start_label.pack()
    time_start = tk.Entry(window)
    time_start.pack()
    time_start.bind("<FocusOut>", get_selected_time_start)

    time_end_label = tk.Label(window, text="End Time:")
    time_end_label.pack()
    time_end = tk.Entry(window)
    time_end.pack()
    time_end.bind("<FocusOut>", get_selected_time_end)

    get_range_button = tk.Button(window, text="Get Date and Time Range", command=get_selected_range)
    get_range_button.pack()

    window.mainloop()

    return datetime_range['start'], datetime_range['end']  # return the stored date-time values



if __name__ == "__main__":
    # Call the function to select the date range
    # Rufe die Funktion auf, um den Datumsumfang auszuwählen


    cur, conn = dbconnect()


    # Übergebe den ausgewählten Datumsumfang an andere Funktionen
    start_datetime_str, end_datetime_str = select_date_time_range()
    print(start_datetime_str)


    word_cloud(cur, start_datetime_str, end_datetime_str)
    plothash(cur, start_datetime_str, end_datetime_str)
    G = networkanalyse(cur, start_datetime_str, end_datetime_str)
    mostengagingtopics(cur, start_datetime_str, end_datetime_str)
    sentimentovertime(cur, start_datetime_str, end_datetime_str)
    trendingtopics(cur, start_datetime_str, end_datetime_str)
    trendingtopicste(cur, start_datetime_str, end_datetime_str)
    toptopics(cur, start_datetime_str, end_datetime_str)


    user_popularity(cur, start_datetime_str, end_datetime_str)
    networkanalyse_follower(cur, start_datetime_str, end_datetime_str)
    analyze_all_users(cur)
    analyze_twitter_users(cur)

    test(cur, start_datetime_str, end_datetime_str)
    cchashtags(cur, start_datetime_str, end_datetime_str)
    pnwords(cur, start_datetime_str, end_datetime_str)
    topic_sentiment_ratio(cur, start_datetime_str, end_datetime_str)
    word_frequency_over_time(cur, 'sec', start_datetime_str, end_datetime_str)


    sentiment(cur, start_datetime_str, end_datetime_str)
    topword(cur, start_datetime_str, end_datetime_str)




    # Datenbankverbindung schließen
    cur.close()
    conn.close()
