import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import mariadb
import json
from wordcloud import WordCloud
from collections import defaultdict
import plotly.graph_objects as go


def dbconnect():
    # Verbindung zur Datenbank herstellen
    conn = mariadb.connect(user='', password='', host="", database="")
    cur = conn.cursor()
    return cur, conn

def plothash(cur):
    # Daten aus der Datenbank abrufen
    cur.execute("SELECT created_at, count, hashtag FROM hashtags GROUP BY created_at, hashtag order by count desc")
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

def networkanalyse(cur):
    G = nx.Graph()
    cur.execute("SELECT text FROM tweets where fetch_id = 1")
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


def mostengagingtopics(cur):
    # First, fetch engagement data and join it with topic data
    cur.execute(
        "SELECT topics.topic_data, tweets.retweet_count, tweets.reply_count, tweets.like_count, tweets.quote_count FROM topics JOIN tweets ON topics.fetch_id = tweets.fetch_id")
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


def sentimentovertime(cur):
    cur.execute("SELECT created_at, positive, neutral, negative FROM sentiments ORDER BY created_at")
    data = cur.fetchall()
    df = pd.DataFrame(data, columns=['created_at', 'positive', 'neutral', 'negative'])
    df.set_index('created_at', inplace=True)
    df.plot()
    plt.show()

def trendingtopics(cur):
    cur.execute("SELECT created_at, count, hashtag FROM hashtags  where count > 2 ORDER BY created_at")
    data = cur.fetchall()
    df = pd.DataFrame(data, columns=['created_at', 'count', 'hashtag'])
    df.set_index('created_at', inplace=True)
    df.groupby('hashtag')['count'].plot()

    df.plot(kind='bar', figsize=(12, 6))
    plt.title("Top Hashtags by Count")
    plt.ylabel("Count")
    plt.show()

def trendingtopicste(cur):
    # Fetch data from the database
    cur.execute("SELECT created_at, topic_data FROM topics ORDER BY created_at")
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

def toptopics(cur):
    # Fetch data from the database
    cur.execute("SELECT topic_data, positive, neutral, negative FROM sentiments JOIN topics ON sentiments.fetch_id = topics.fetch_id")
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


def topword(cur):
    cur.execute("SELECT word, count FROM words where word <> 'crypto'")
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

def sentiment(cur):
    cur.execute(
        "SELECT t.hashtag, s.positive, s.neutral, s.negative FROM hashtags as t JOIN sentiments as s ON t.fetch_id = s.fetch_id where t.count > 10 and t.hashtag <> '#crypto' order by t.created_at desc")
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


def word_cloud(cur):
    cur.execute("SELECT word FROM words")
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

def pnwords(cur):
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from collections import Counter
    import nltk

    nltk.download('vader_lexicon')

    sia = SentimentIntensityAnalyzer()
    cur.execute("SELECT text FROM tweets")
    data = cur.fetchall()

    # Assuming 'text' column contains the tweet text
    all_words = ' '.join([word[0] for word in data]).split()
    sentiment_scores = {word: sia.polarity_scores(word)['compound'] for word in all_words}

    positive_words = [word for word, score in sentiment_scores.items() if score > 0]
    negative_words = [word for word, score in sentiment_scores.items() if score < 0]

    print('Positive words:', Counter(positive_words).most_common(10))
    print('Negative words:', Counter(negative_words).most_common(10))

def test(cur):
    G = nx.Graph()
    hashtag_counts = defaultdict(int)
    hashtag_pairs = defaultdict(int)
    cur.execute("SELECT text FROM tweets")
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



def cchashtags(cur):


    G = nx.Graph()
    hashtag_counts = defaultdict(int)
    hashtag_pairs = defaultdict(int)
    cur.execute("SELECT text FROM tweets")
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


def word_cloud(cur):
    cur.execute("SELECT word FROM words")
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

def topic_sentiment_ratio(cur):
    cur.execute("SELECT topic_data, positive, negative FROM sentiments JOIN topics ON sentiments.fetch_id = topics.fetch_id")
    data = cur.fetchall()

    df = pd.DataFrame.merge

def word_frequency_over_time(cur, word):
    # Fetch data from the database
    cur.execute(f"SELECT created_at, count FROM words WHERE word = '{word}' ORDER BY created_at")
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

if __name__ == "__main__":
    cur, conn = dbconnect()
    #test(cur)
    #cchashtags(cur)
    pnwords(cur)
    #topic_sentiment_ratio(cur)
    word_frequency_over_time(cur, 'pepe')
    word_cloud(cur)
    trendingtopicste(cur)
    sentiment(cur)
    topword(cur)

    toptopics(cur)
    sentimentovertime(cur)
    plothash(cur)
    mostengagingtopics(cur)
    G = networkanalyse(cur)
    # Datenbankverbindung schließen
    cur.close()
    conn.close()
