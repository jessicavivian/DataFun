import pandas as pd
from textblob import TextBlob

# Load the data
df = pd.read_csv('datasets/test_comments.csv')

# Print first line of the data
print(df.head(1))

# Calculate sentiment polarity
df['polarity'] = df['comment text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Classify sentiments as positive, neutral, or negative
df['sentiment'] = df['polarity'].apply(lambda x: 'positive' if x > 0 else ('neutral' if x==0 else 'negative'))

# Print result
print(df[['comment text', 'sentiment']].head(10))