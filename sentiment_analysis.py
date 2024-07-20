import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import pandas as pd

df = pd.read_csv(r"data/BA_reviews.csv")

# df = df.dropna(subset=["date"])
# df['date'] = df['date'].astype('datetime64[ns]')
df['reviews'] = df['reviews'].astype(str)
# df['state'] = df['state'].astype('category')

print(df.head)

analyzer = SIA()
df['compound'] = [analyzer.polarity_scores(x)['compound'] for x in df['reviews']]
df['negative'] = [analyzer.polarity_scores(x)['neg'] for x in df['reviews']]
df['neutral'] = [analyzer.polarity_scores(x)['neu'] for x in df['reviews']]
df['positive'] = [analyzer.polarity_scores(x)['pos'] for x in df['reviews']]

#create columns
df['sentiment']='neutral'
df.loc[df.compound>0.95,'sentiment']='positive'
df.loc[df.compound<-0.05,'sentiment']='negative'
df.head()

#on-screen summary
print(df['sentiment'].value_counts()['neutral'])
print(df['sentiment'].value_counts()['positive'])
print(df['sentiment'].value_counts()['negative'])

#save to csv file
df.to_csv(r"data/sentiment_data.csv", index=False)