import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, time, timedelta
import mariadb
import pandas as pd
from plotly.subplots import make_subplots

from wordcloud import WordCloud
import io
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from gensim import models, corpora
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.ndimage import uniform_filter1d

import base64
import numpy as np

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=['assets/style.css'])



def dbconnect():
    # Verbindung zur Datenbank herstellen
    conn = mariadb.connect(user='j0n1', password='12345', host="192.168.178.89", database="test")
    cur = conn.cursor()
    return cur, conn


def textmining(cur, start_datetime, end_datetime, stopwords):
    cur.execute(f"SELECT created_at, text FROM tweets WHERE created_at BETWEEN '{start_datetime}' AND '{end_datetime}'")
    data = cur.fetchall()
    texts = [tweet[1] for tweet in data]

    # Tokenization and filtering of texts
    all_words = []
    for text in texts:
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in stopwords]
        all_words.append(words)

    # Perform topic modeling
    dictionary = corpora.Dictionary(all_words)
    corpus = [dictionary.doc2bow(words) for words in all_words]
    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary)

    # Get the top 5 topics for each document
    topic_distributions = []
    for text in all_words:
        doc_bow = dictionary.doc2bow(text)
        topic_distribution = lda_model[doc_bow]
        topic_distributions.append(topic_distribution)

    # Extract the top topics over time
    interval = timedelta(hours=1)
    start_time = pd.to_datetime(start_datetime)
    end_time = pd.to_datetime(end_datetime)
    time_range = pd.date_range(start=start_time, end=end_time, freq=interval)
    topic_counts_over_time = [[] for _ in range(5)]

    for i in range(len(time_range) - 1):
        start_index = time_range[i]
        end_index = time_range[i + 1]
        topic_counts = [0] * 5  # Initialize topic counts for each time interval
        for j, topic_dist in enumerate(topic_distributions):
            tweet_time = pd.to_datetime(data[j][0])
            if start_index <= tweet_time < end_index:
                sorted_topics = sorted(topic_dist, key=lambda x: x[1], reverse=True)
                for k in range(5):
                    if k < len(sorted_topics):
                        topic_id = sorted_topics[k][0]
                        topic_words = [word[0] for word in lda_model.show_topic(topic_id, topn=5)]
                        if topic_id == k:
                            topic_counts[k] += 1  # Increment topic count for the current topic
                    else:
                        break  # No more topics to consider

        for k in range(5):
            topic_counts_over_time[k].append(topic_counts[k])

    # Create line chart for top topics over time
    topic_labels = [
        f"Topic {i + 1}: {', '.join(word[0] for word in lda_model.show_topic(i, topn=5))}"
        for i in range(5)
    ]

    fig = go.Figure()
    for i, topic_counts in enumerate(topic_counts_over_time):
        fig.add_trace(
            go.Scatter(x=list(range(len(topic_counts))), y=topic_counts,
                       mode='lines+markers', name=topic_labels[i], line=dict(width=2))
        )

    fig.update_layout(
        title='Top 5 Topics Over Time',
        title_font=dict(size=24, family='Arial'),
        xaxis=dict(title='Time Interval', tickangle=45, tickfont=dict(size=10),
                   tickvals=list(range(len(topic_counts_over_time[0]))),
                   ticktext=[f"{start_time} - {end_time}" for start_time, end_time in zip(time_range[:-1], time_range[1:])]),
        yaxis=dict(title='Count', tickfont=dict(size=12)),
        width=800,
        height=600
    )

    return {'figure': fig}




def get_tweet_count(cur, start, end):
    cur.execute(f"SELECT count(id) FROM tweets where created_at between '{start}' and '{end}'")
    tweet_count = cur.fetchone()[0]
    return tweet_count
def network_analyze(cur, start, end):
    # ... code for retrieving tweets and constructing the network graph ...
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

    pos = nx.spring_layout(G, k=0.15)

    # Calculate the count of each hashtag
    hashtag_counts = {node: sum([1 for edge in G.edges(node)] + [1 for edge in G.edges(None, node)]) for node in
                      G.nodes()}

    # Adjust the size of the bubbles based on the hashtag count
    node_sizes = [hashtag_counts[node] for node in G.nodes()]
    min_size = min(node_sizes)
    max_size = max(node_sizes)
    scaling_factor = 100

    scaled_sizes = [(size - min_size) * scaling_factor / (max_size - min_size) + 5 for size in node_sizes]

    nodes = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode="markers",
        hoverinfo="text",
        marker=dict(
            color="blue",
            size=scaled_sizes,
            line=dict(color="black", width=0.5)
        ),
        text=[node for node in G.nodes()]
    )

    # Calculate the edge coordinates
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

    edges = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(color="grey", width=0.5)
    )

    layout = go.Layout(
        title="Network Analysis",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    fig = go.Figure(data=[edges, nodes], layout=layout)

    return fig

def plothash_tag(cur, start, end, stopwords):
    cur.execute(f"SELECT text, created_at FROM tweets WHERE created_at BETWEEN '{start}' AND '{end}'")
    data = cur.fetchall()

    # Create a DataFrame with the fetched data
    df = pd.DataFrame(data, columns=['text', 'created_at'])
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Count the occurrences of each word
    word_counts = df['text'].str.lower().str.split().explode().apply(
        lambda word: word if (word.startswith('#') and word not in stopwords) else np.nan
    ).dropna().value_counts()

    # Scale the word counts between 0 and 1
    max_count = word_counts.max()
    word_counts_scaled = word_counts / max_count

    # Create a pandas DataFrame with word counts
    df_counts = pd.DataFrame({'word': word_counts_scaled.index, 'count': word_counts_scaled.values})

    # Select the top 20 words based on counts
    top_words = df_counts.sort_values('count', ascending=False).nlargest(20, 'count')['word']

    # Create the line chart
    fig = make_subplots(rows=1, cols=1)

    for word in top_words:
        # Create a dataframe for each word
        word_df = df[df['text'].str.contains(word, regex=False)].copy()

        # Set 'created_at' as the index before resampling
        word_df.set_index('created_at', inplace=True)

        # Group by word and resample the data to a regular time interval
        word_df_resampled = word_df.resample('15Min').count()

        # Apply moving average with a window size of 4 (equivalent to 1 hour)
        window_size = 4
        word_df_resampled_smoothed = word_df_resampled.rolling(window=window_size, min_periods=1, center=True).mean()

        # Convert the smoothed array back to a DataFrame
        word_df_resampled_smoothed = pd.DataFrame(word_df_resampled_smoothed, columns=['text'])

        fig.add_trace(go.Scatter(
            x=word_df_resampled_smoothed.index,
            y=word_df_resampled_smoothed['text'],
            mode="lines",
            name=word,
            marker=dict(symbol="x")
        ))

    fig.update_layout(
        title="Word Occurrences Over Time (Smoothed)",
        xaxis=dict(title="Time"),
        yaxis=dict(title="Count")
    )

    return fig
def plothash(cur, start, end, stopwords):
    cur.execute(f"SELECT text, created_at FROM tweets WHERE created_at BETWEEN '{start}' AND '{end}'")
    data = cur.fetchall()

    # Create a DataFrame with the fetched data
    df = pd.DataFrame(data, columns=['text', 'created_at'])
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Count the occurrences of each word
    word_counts = df['text'].str.lower().str.split().explode().apply(
        lambda word: word if word not in stopwords else 'ddd').value_counts()

    # Scale the word counts between 0 and 1
    max_count = word_counts.max()
    word_counts_scaled = word_counts / max_count

    # Create a pandas DataFrame with word counts
    df_counts = pd.DataFrame({'word': word_counts_scaled.index, 'count': word_counts_scaled.values})

    # Select the top 20 words based on counts
    top_words = df_counts.sort_values('count', ascending=False).nlargest(20, 'count')['word']

    # Create the line chart
    fig = make_subplots(rows=1, cols=1)

    for word in top_words:
        # Create a dataframe for each word
        word_df = df[df['text'].str.contains(word, regex=False)].copy()

        # Set 'created_at' as the index before resampling
        word_df.set_index('created_at', inplace=True)

        # Group by word and resample the data to a regular time interval
        word_df_resampled = word_df.resample('15Min').count()

        # Apply moving average with a window size of 4 (equivalent to 1 hour)
        window_size = 4
        word_df_resampled_smoothed = word_df_resampled.rolling(window=window_size, min_periods=1, center=True).mean()

        # Convert the smoothed array back to a DataFrame
        word_df_resampled_smoothed = pd.DataFrame(word_df_resampled_smoothed, columns=['text'])

        fig.add_trace(go.Scatter(
            x=word_df_resampled_smoothed.index,
            y=word_df_resampled_smoothed['text'],
            mode="lines",
            name=word,
            marker=dict(symbol="x")
        ))

    fig.update_layout(
        title="Word Occurrences Over Time (Smoothed)",
        xaxis=dict(title="Time"),
        yaxis=dict(title="Count")
    )

    return fig

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
    stop_words.update(['https', 'co', 'rt', 'u'])

    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stop_words,
                          min_font_size=10).generate(text_filtered.lower())

    plt.figure(figsize=(6, 6), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)


    # Save the plot to a BytesIO buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    # Encode the plot image as a base64 string
    image_base64 = base64.b64encode(buffer.read()).decode()

    return image_base64

def network_analyze_follower(cur, start, end):
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

    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G, k=2.15)
    # Calculate the node sizes based on weight. Adjust the size calculation to your preference.
    node_sizes = [np.sqrt(np.abs(G.degree(node, weight='weight')) * 1) * 2 for node in G.nodes()]

    nx.draw_networkx(G, pos, node_size=node_sizes, node_color='blue', font_size=10, edge_color='grey')

    # Plot the graph using Plotly
    pos = nx.spring_layout(G, k=2.15)
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

    nodes = go.Scatter(
        x=[pos[node][0] for node in G.nodes()],
        y=[pos[node][1] for node in G.nodes()],
        mode="markers",
        hoverinfo="text",
        marker=dict(
            color="blue",
            size=[np.sqrt(np.abs(G.degree(node, weight='weight')) * 1) * 2 for node in G.nodes()],
            sizemode='diameter',  # Use 'diameter' for direct size specification
            sizeref=0.05,  # Adjust this value to control the size scaling
            line=dict(color="black", width=0.5)
        ),
        text=[node for node in G.nodes()]
    )

    edges = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(color="grey", width=0.5)
    )

    layout = go.Layout(
        title="Network Analysis (Followers > 500)",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=40, l=40, r=40, t=40),
        height=800,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )

    fig = go.Figure(data=[edges, nodes], layout=layout)

    return G, fig


app.layout = html.Div(
    [
        html.H1("Select Date and Time Range"),

        html.Div(
            [
                dcc.DatePickerSingle(
                    id="start-date-picker",
                    placeholder="Start Date",
                    display_format="YYYY-MM-DD",
                    style={"marginRight": "10px"},
                    className="date-picker",
                    date=datetime.now().date()
                ),
                dcc.DatePickerSingle(
                    id="end-date-picker",
                    placeholder="End Date",
                    display_format="YYYY-MM-DD",
                    style={"marginRight": "10px"},
                    className="date-picker",
                    date=datetime.now().date()
                ),
                html.Div(
                    [
                        dcc.Input(
                            id="start-time-input",
                            type="text",
                            placeholder="Start Time",
                            style={"marginRight": "10px"},
                            value="00:00"
                        ),
                        dcc.Input(
                            id="end-time-input",
                            type="text",
                            placeholder="End Time",
                            style={"marginRight": "10px"},
                            value="23:59"
                        ),
                    ],
                    className="date-picker"
                ),
                html.Button(
                    "Get Date and Time Range",
                    id="get-range-button",
                    n_clicks=0,
                    style={"marginTop": "10px"}
                ),
            ],
            style={"marginBottom": "20px"}
        ),

        dcc.Store(id="date-time-store", data={}),

        html.Div(
            [
                html.Div(id="tweet-count-container"),
                html.Table(
                    [
                        html.Tbody(
                            [
                                html.Tr(
                                    [
                                        html.Td(html.Div(id="network-graph-container")),
                                        html.Td(html.Div(id="hashtag-plot-container"))
                                    ]
                                ),
                                html.Tr(
                                    [
                                        html.Td(html.Div(id="sentiment-plot-container")),
                                        html.Td(html.Div(id="word-cloud-container"))
                                    ]
                                )
                            ]
                        )
                    ],
                    style={"width": "100%"}
                ),

            ],
            className="plot-container"
        ),

        dcc.Graph(id="network-graph-container-fol"),
        html.Div(children=[
        dcc.Graph(id="topics")
    ]),
    html.Div(id="hashtag-plot-container-tag")
    ],
    className="container"
)

@app.callback(
    Output("date-time-store", "data"),
    Input("get-range-button", "n_clicks"),
    State("start-date-picker", "date"),
    State("end-date-picker", "date"),
    State("start-time-input", "value"),
    State("end-time-input", "value"),
    State("date-time-store", "data")
)

def update_date_time_store(n_clicks, start_date, end_date, start_time, end_time, stored_data):
    if n_clicks > 0:
        start_datetime = datetime.combine(datetime.strptime(start_date, "%Y-%m-%d").date(), time.fromisoformat(start_time))
        end_datetime = datetime.combine(datetime.strptime(end_date, "%Y-%m-%d").date(), time.fromisoformat(end_time))
        new_data = {
            "start_datetime": start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
            "end_datetime": end_datetime.strftime("%Y-%m-%d %H:%M:%S")
        }
        return new_data
    else:
        return stored_data

@app.callback(
    Output("network-graph-container", "children"),
    Input("date-time-store", "data")
)
def update_network_graph(data):
    if data and "start_datetime" in data and "end_datetime" in data:
        cur, conn = dbconnect()
        start_datetime = data["start_datetime"]
        end_datetime = data["end_datetime"]
        graph = network_analyze(cur, start_datetime, end_datetime)
        conn.close()
        return dcc.Graph(id="network-graph", figure=graph)
    else:
        return html.Div()

@app.callback(
    Output("hashtag-plot-container", "children"),
    Input("date-time-store", "data")
)
def update_hashtag_plot(data):
    if data and "start_datetime" in data and "end_datetime" in data:
        cur, conn = dbconnect()
        start_datetime = data["start_datetime"]
        end_datetime = data["end_datetime"]
        stop_words_english = set(stopwords.words('english'))
        stop_words_german = set(stopwords.words('german'))
        stop_words = stop_words_english.union(stop_words_german)
        stop_words.update(['https', 'co', 'rt', '&amp;', '-', 'get'])
        plot = plothash(cur, start_datetime, end_datetime, stop_words)
        conn.close()
        return dcc.Graph(id="hashtag-plot", figure=plot)
    else:
        return html.Div()
@app.callback(
    Output("sentiment-plot-container", "children"),
    Input("date-time-store", "data")
)
def update_sentiment_plot(data):
    if data and "start_datetime" in data and "end_datetime" in data:
        cur, conn = dbconnect()
        start_datetime = data["start_datetime"]
        end_datetime = data["end_datetime"]
        plot = sentimentovertime(cur, start_datetime, end_datetime)
        conn.close()
        return dcc.Graph(id="sentiment-plot", figure=plot)
    else:
        return html.Div()


def sentimentovertime(cur, start, end):
    cur.execute(
        f"SELECT text, created_at FROM tweets WHERE created_at BETWEEN '{start}' AND '{end}' ORDER BY created_at"
    )
    data = cur.fetchall()
    df = pd.DataFrame(data, columns=['text', 'created_at'])
    df['created_at'] = pd.to_datetime(df['created_at'])  # Convert 'created_at' column to datetime
    df.set_index('created_at', inplace=True)

    # Initialize sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()

    # Calculate sentiments for every hour
    sentiments = {"positive": [], "neutral": [], "negative": []}
    for timestamp, tweet_text in zip(df.index, df['text']):
        sentiment_scores = analyzer.polarity_scores(tweet_text)
        compound_score = sentiment_scores['compound']

        # Classify the sentiment
        if compound_score >= 0.05:
            sentiments["positive"].append(1)
            sentiments["neutral"].append(0)
            sentiments["negative"].append(0)
        elif compound_score > -0.05 and compound_score < 0.05:
            sentiments["positive"].append(0)
            sentiments["neutral"].append(1)
            sentiments["negative"].append(0)
        else:
            sentiments["positive"].append(0)
            sentiments["neutral"].append(0)
            sentiments["negative"].append(1)

    # Create a DataFrame with sentiments and resample to 1-hour intervals
    df_sentiments = pd.DataFrame(sentiments, index=df.index)
    df_resampled = df_sentiments.resample('1H').sum()

    # Calculate the moving average using a window of 1 hour
    df_smoothed = df_resampled.rolling('1H').mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_smoothed.index, y=df_smoothed['positive'], mode='lines', name='Positive', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df_smoothed.index, y=df_smoothed['neutral'], mode='lines', name='Neutral', line=dict(color='gray')))
    fig.add_trace(go.Scatter(x=df_smoothed.index, y=df_smoothed['negative'], mode='lines', name='Negative', line=dict(color='red')))

    return fig

@app.callback(
    Output("word-cloud-container", "children"),
    Input("date-time-store", "data")
)
def update_word_cloud(data):
    if data and "start_datetime" in data and "end_datetime" in data:
        cur, conn = dbconnect()
        start_datetime = data["start_datetime"]
        end_datetime = data["end_datetime"]
        image_base64 = word_cloud(cur, start_datetime, end_datetime)
        conn.close()
        return html.Img(src=f"data:image/png;base64,{image_base64}")

    return html.Div()


@app.callback(
    Output("network-graph-container-fol", "figure"),
    Input("date-time-store", "data")
)
def update_network_graph_fol(data):
    if data and "start_datetime" in data and "end_datetime" in data:
        cur, conn = dbconnect()
        start_datetime = data["start_datetime"]
        end_datetime = data["end_datetime"]
        nx_graph, graph = network_analyze_follower(cur, start_datetime, end_datetime)
        conn.close()

        pos = nx.spring_layout(nx_graph, k=40)
        edge_x = []
        edge_y = []
        for edge in nx_graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        node_sizes = [np.sqrt(np.abs(nx_graph.degree(node, weight='weight'))) * 10 for node in nx_graph.nodes()]

        nodes = go.Scatter(
            x=[pos[node][0] for node in nx_graph.nodes()],
            y=[pos[node][1] for node in nx_graph.nodes()],
            mode="markers",
            hoverinfo="text",
            marker=dict(
                color="blue",
                size=node_sizes,
                sizemode="diameter",  # Use "diameter" for direct size specification
                sizeref=100,  # Adjust this value to control the size scaling
                line=dict(color="black", width=0.5)
            ),
            text=[node for node in nx_graph.nodes()]
        )

        edges = go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="grey", width=0.5)
        )

        layout = go.Layout(
            title="Network Analysis (Followers > 500)",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=1000,
            width=1000
        )

        fig = go.Figure(data=[edges, nodes], layout=layout)

        return fig
    else:
        return go.Figure()

@app.callback(
    Output("tweet-count-container", "children"),
    Input("date-time-store", "data")
)
def update_tweet_count(data):
    # Perform the necessary calculations to get the tweet count
    # Replace the following line with your actual tweet count calculation
    cur, conn = dbconnect()
    start_datetime = data["start_datetime"]
    end_datetime = data["end_datetime"]
    tweet_count = get_tweet_count(cur, start_datetime, end_datetime)
    conn.close()


    return html.H3(f"Number of Analyzed Tweets: {tweet_count}")

@app.callback(
    Output("topics", 'figure'),
    Input("date-time-store", "data")
)
def update_textmining(data):
    cur, conn = dbconnect()
    start_datetime = data["start_datetime"]
    end_datetime = data["end_datetime"]
    stop_words_english = set(stopwords.words('english'))
    stop_words_german = set(stopwords.words('german'))
    stop_words = stop_words_english.union(stop_words_german)
    stop_words.update(['https', 'co', 'rt', '&amp;', '-', 'get', 'u'])
    topics = textmining(cur, start_datetime, end_datetime, stop_words)
    conn.close()
    return topics['figure']

@app.callback(
    Output("hashtag-plot-container-tag", "children"),
    Input("date-time-store", "data")
)
def update_hashtag_plot_tag(data):
    if data and "start_datetime" in data and "end_datetime" in data:
        cur, conn = dbconnect()
        start_datetime = data["start_datetime"]
        end_datetime = data["end_datetime"]
        stop_words_english = set(stopwords.words('english'))
        stop_words_german = set(stopwords.words('german'))
        stop_words = stop_words_english.union(stop_words_german)
        stop_words.update(['https', 'co', 'rt', '&amp;', '-', 'get'])
        plot = plothash_tag(cur, start_datetime, end_datetime, stop_words)
        conn.close()
        return dcc.Graph(id="hashtag-plot-tag", figure=plot)
    else:
        return html.Div()


if __name__ == "__main__":
    app.run_server(debug=True)
