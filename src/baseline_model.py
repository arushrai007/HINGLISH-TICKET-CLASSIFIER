import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

train_df=pd.read_csv('data/train.csv')
test_df=pd.read_csv('data/test.csv')

vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2)) #convert texts to numbers so that ml model could understand is what vectorization is
#tf-idf term frequency inversedocument frequency it basically tells us how important a word is to a document in a collection or corpus. It is often used as a weighting factor in information retrieval and text mining.
#max features=3000 means we are only going to take the top 3000 features from the text data. ngram_range=(1, 2) means we are going to take unigrams and bigrams from the text data.
X_train = vectorizer.fit_transform(train_df['message'])
X_test = vectorizer.transform(test_df['message'])
y_train = train_df['label_id']
y_test = test_df['label_id']


clf=LogisticRegression(max_iter=1000, class_weight='balanced') #MAX_TER REFERS TO THE MAXIMUM NUMBER OF ITERATIONS FOR THE SOLVER TO CONVERGE. CLASS_WEIGHT='BALANCED' MEANS THAT THE ALGORITHM WILL ADJUST THE WEIGHTS OF THE CLASSES INVERSELY PROPORTIONAL TO THEIR FREQUENCY IN THE DATASET. THIS IS USEFUL WHEN DEALING WITH IMBALANCED DATASETS.
clf.fit(X_train,y_train)

pred=clf.predict(X_test)

print("Classification Report(Logistic Regression+TF-IDF):")
print(classification_report(y_test, pred))

