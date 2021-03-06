import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics

#Loading the data set - training data.
df = pd.read_csv('../../../../data/data_for_ml.csv', sep=',')

X_train = df['utterances']
Y_train = df['intent']

print(X_train.shape)

df = pd.read_csv('../../../../data/test_data.csv', sep=',')

X_test = df['utterances']
Y_test = df['intent']

print(X_test)

"""
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
# print(X_train_counts.shape)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# print(X_train_tfidf.shape)

# Machine Learning
# Training Naive Bayes (NB) classifier on training data.
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, Y_train)

"""

# Building a pipeline: We can write less code and do all of the above, by building a pipeline as follows:
# The names ‘vect’ , ‘tfidf’ and ‘clf’ are arbitrary but will be used later.
# We will be using the 'text_clf' going forward.
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
text_clf = text_clf.fit(X_train, Y_train)

# print("\n\nPrediction using MultinomialNB: >> ", clf.predict(count_vect.transform(X_test)))
# print("Prediction using Pipeline: >> ", text_clf.predict(X_test))

# Performance of MultinomialNB Classifier
import numpy as np
predicted = text_clf.predict(X_test)
# print("Performance of MultinomialNB Classifier: >> ", np.mean(predicted == Y_test))
# print(metrics.classification_report(Y_test, predicted, target_names=Y_test))

# Training Support Vector Machines - SVM and calculating its performance
from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42))])

text_clf_svm = text_clf_svm.fit(X_train, Y_train)
predicted_svm = text_clf_svm.predict(X_test)
print("\n\nSVM Prediction: >>> ", predicted_svm)
print("SVM Performance: >>> ", np.mean(predicted_svm == Y_test))
print(metrics.classification_report(Y_test, predicted_svm, target_names=Y_test))

# In[18]:

# Grid Search
# Here, we are creating a list of parameters for which we would like to do performance tuning.
# All the parameters name start with the classifier name (remember the arbitrary name we gave).
# E.g. vect__ngram_range; here we are telling to use unigram and bigrams and choose the one which is optimal.

from sklearn.model_selection import GridSearchCV
parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}


# In[19]:

# Next, we create an instance of the grid search by passing the classifier, parameters
# and n_jobs=-1 which tells to use multiple cores from user machine.

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(X_train, Y_train)


# In[23]:

# To see the best mean score and the params, run the following code

gs_clf.best_score_
gs_clf.best_params_

# Output for above should be: The accuracy has now increased to ~90.6% for the NB classifier (not so naive anymore! 😄)
# and the corresponding parameters are {‘clf__alpha’: 0.01, ‘tfidf__use_idf’: True, ‘vect__ngram_range’: (1, 2)}.


# In[24]:

# Similarly doing grid search for SVM
from sklearn.model_selection import GridSearchCV
parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf-svm__alpha': (1e-2, 1e-3)}

gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(X_train, Y_train)


print("gs_clf_svm.best_score_: >> ", gs_clf_svm.best_score_)
print("gs_clf_svm.best_params_: >> ", gs_clf_svm.best_params_)


# In[25]:

# NLTK
# Removing stop words
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])


# In[26]:

# Stemming Code

import nltk
# Already downloaded, so commented below line
# nltk.download()

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()),
                             ('mnb', MultinomialNB(fit_prior=False))])

text_mnb_stemmed = text_mnb_stemmed.fit(X_train, Y_train)

predicted_mnb_stemmed = text_mnb_stemmed.predict(X_test)
print('\n\npredicted_mnb_stemmed: >> ', predicted_mnb_stemmed)
print(np.mean(predicted_mnb_stemmed == Y_test))
# print(metrics.classification_report(Y_test, predicted_mnb_stemmed, target_names=Y_test))
# score = metrics.accuracy_score(Y_test, predicted_mnb_stemmed)
# print("Accuracy: %f" % score)


# In[ ]:
