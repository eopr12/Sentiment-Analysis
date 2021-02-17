import nltk
nltk.download('wordnet')
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

## Figure1 ##
# Create quick lambda functions to find the polarity of each review
# Terminal / Anaconda Navigator: conda install -c conda-forge textblobfrom textblob import TextBlob
df['Text']= df['Text'].astype(str) #Make sure about the correct data type
pol = lambda x: TextBlob(x).sentiment.polarity
df['polarity'] = df['Text'].apply(pol) # depending on the size of your data, this step may take some time.


num_bins = 50
plt.figure(figsize=(10,6))
n, bins, patches = plt.hist(df.polarity, num_bins, facecolor='blue', alpha=0.5)
plt.xlabel('Polarity')
plt.ylabel('Number of Reviews')
plt.title('Histogram of Polarity Score')
plt.show();