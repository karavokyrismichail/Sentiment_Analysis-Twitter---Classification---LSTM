import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# for shaping the texts
nltk.download('stopwords')
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# loading the dataset
df = pd.read_csv('Data/training.1600000.processed.noemoticon.csv', encoding = 'Latin-1', names=('target','id','date','flag','username','tweet'))

# print the features of dataset
print(df.head())
print(df.info())
# sns.countplot(x = 'target',data = df)
# plt.show()

# keep only target and text features
df.drop(['id','date','flag','username'], axis=1, inplace=True)
# getting a sample of param% of the dataset
df = df.sample(frac = 0.1)

def text_preprocessing(text):

    # text cleaning
    sentence = re.sub(r'[^\w\s]', ' ',text )
    sentence = re.sub(r'[0-9]', '', sentence)
    
    # tokenize
    words = nltk.word_tokenize(sentence)
    for word in words:
            word.lower()
    
    # remove stopwords
    words = [w for w in words if not w in stop_words]
    
    # stemming
    words = [stemmer.stem(w) for w in words]
    
    # lemmatizing
    final_words = [lemmatizer.lemmatize(w) for w in words]
    
    return final_words

# creating vectors of words for each tweet
df['tweet'] = df.apply(lambda x: text_preprocessing(x['tweet']), axis=1)

# convert 4 to 1 as the positive tag
df['target'] = df['target'].apply(lambda x: 1 if x==4 else x)
# creating list of the cleared tweets
string_list = []
for i in df.tweet :
    str1 = " " 
    string = str1.join(i)
    string_list.append(string)
df['tweet'] = string_list


X = df['tweet']
y = df['target']
# split the sampled dataset 70-30
raw_X_train, raw_X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42) # 42-> life, the universe, everything

# tweet data to vector matrix using tf-idf
vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000, lowercase= False)
vectoriser.fit(raw_X_train)

# creating document-term matrix 
X_train = vectoriser.transform(raw_X_train)
X_test  = vectoriser.transform(raw_X_test)
# returns all the data we want 
def getData():
    return X_train, X_test, y_train, y_test, raw_X_train, raw_X_test