import re
import math
import numpy as np
import pandas as pd
import general_functions as lf
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

def parser(title):
    
    '''Removes any non-alphabetical characters, converts to lower case, and lemmatizes each word in a document'''
    
    wl = WordNetLemmatizer()
    letters = re.sub('[^a-zA-Z]', ' ', title)
    letters = letters.lower()
    words = re.split('\s+', letters)
    words = [wl.lemmatize(x) for x in words]
    return (' '.join(words))

def create_feature_df(df, vectorizer):
    
    '''Applies a CountVectorizer transformation to the data, then returns a list of dataframes of the counted words, one for each
    subreddit'''
    
    features = vectorizer.fit_transform(df['ptitle'])
    frame = pd.DataFrame(features.toarray(), columns=vectorizer.get_feature_names())
    frame['sub'] = df['sub']
    sub_list = []
    
    for x in range(3):
        sub_list.append(frame[frame['sub']==x].copy())
    
    return sub_list

def top_lists(frame, vect, num):
    
    '''Function which receives a feature dataframe, the vectorizer that created that dataframe, and the number of top words desired
    Returns two lists: the top words and the corresponding frequency of those words in the data set'''
    
    return list(lf.top_words(frame, vect, num).keys()), list(lf.top_words(frame, vect, num).values())

def plot_top(data, vectorizer, num, title, hsize='default'):
    
    '''Function which receives a list of dataframes, a vectorizer, the number of top words desired, and some plot paramaters
    Runs the top_lists function on each of the dataframes in the data list, then plots all of the top words for each of the frames as bar charts'''
    
    top_word_list = []
    top_counts = []
    
    for words in data:
        x, y = top_lists(words, vectorizer, num)
        y = [math.floor(num) for num in y]
        top_word_list.append(x[::-1])
        top_counts.append(y[::-1])
        
    lf.subplot(['barh']*3, x_list=[range(num)]*3, title_list=['Magic', 'Eternal', 'Hearthstone'], xlab_list=['Count']*3, ylab_list=top_word_list, y_list=top_counts, fig_title=title, h_size=hsize, bar_numbers=True)
    return top_word_list, top_counts

# Function which searches for the presence of a word in a submission and then captures the number of comments associated with that post
# Returns a list of the total number of comments associated with that word for each subreddit

def word_comments(frame, features, word):
    
    '''Function which searches for the presence of a word in a submission and then captures the number of comments associated with that post
    Returns a list of the total number of comments associated with that word for each subreddit'''
    
    word_comment_list = []
    for x in range(3):
        words = frame.iloc[features[x][features[x][word] > 0].index]
        if len(words) == 0:
            word_comment_list.append(0)
        else:
            word_comment_list.append(round(words['num_comments'].sum()/len(words),1))
    return word_comment_list

# Function which creates a classifier of the chosen type, an integer corresponding to the game of interest, and the target train/test lists (y)
# Creates a classifier, fits the model, and returns lists of the features, list of corresponding coefficients/importances, model test score, and the fitted model instance
# Contains two subfunctions, as models either return coefficients or feature_importance as their output
# Model dictionary can handle Random Forest, Multinomial Naive Bayes, Logistic Regression, Linear SVM, Gradient Boost, and Ada Boost Classifiers

def onevs(classifier, tcgid, x, y):
    
    '''Function which applies a classifier model of the passed type for the subreddit of interest (tcgid)
    The training x and y data is also passed so that the model can be fit
    Returns the list of coefficients or feature importances as well as the fitted model instance
    Contains two subfunctions, since the models either generate coefficient or feature importance values, which are handled differently
    Currently supports Random Forest, Multinomial Naive Bayes, Logistic Regression, Linear SVM, Gradient Boost and ADA Boost classifiers'''
    
    tcgs = ['Magic','Eternal','Hearthstone']
    
    def coef_dict(classifier, x, y ):
        
        model = {'Linear SVM':SVC(kernel='linear', random_state=42),
                 'Logistic Regression':LogisticRegressionCV(solver='liblinear', max_iter=1000),
                 'Naive Bayes':MultinomialNB()}
        mod = model[classifier]
        mod.fit(x, y)
        if classifier == 'Linear SVM':
            modco = mod.coef_.toarray()
        else:
            modco = mod.coef_
        coefs = [-modco[0][z] for z in range(x.shape[1])]
        return coefs, mod
    
    def imp_dict(classifier, x, y):
        model = {'Random Forest':RandomForestClassifier(random_state=42),
                 'Gradient Boost':GradientBoostingClassifier(),
                 'Ada Boost':AdaBoostClassifier()}
        mod = model[classifier]
        mod.fit(x, y)
        return mod.feature_importances_, mod

    models = {'Random Forest':imp_dict,
              'Naive Bayes':coef_dict,
              'Logistic Regression':coef_dict,
              'Linear SVM':coef_dict,
              'Gradient Boost':imp_dict,
              'Ada Boost':imp_dict}
    
    one_train = np.where(y == tcgid, 0 ,1)
    
    coefs, mod = models[classifier](classifier, x, one_train)
    
    return coefs, mod

def simple_grid_top(grid, steps, values='coef'):
    
    '''Creates a dictionary from a gridsearched simple pipeline, which consists of only two steps: a countvectorizer and a classifier model
    Steps argument are the key names for the pipe steps as a list [model, countvectorizer]
    Values by default='coef' for models that return coefficient values. Set values to anything else will instead return feature importances
    Returns a dataframe where the index are the feature names'''
    
    if values == 'coef':
        df = pd.DataFrame(grid.best_estimator_.named_steps[steps[0]].coef_, columns=grid.best_estimator_.named_steps[steps[1]].get_feature_names(), index=['Magic','Eternal','Hearthstone']).transpose()
        top_df = pd.DataFrame(columns=['Magic','Eternal','Hearthstone'])
        for x in ['Magic','Eternal','Hearthstone']:
            top_df[x] = df.sort_values(by=x, ascending=False).index
            top_df[x+' coef'] = list(df.sort_values(by=x, ascending=False)[x])
    else:
        top_df = pd.DataFrame(grid.best_estimator_.named_steps[steps[0]].feature_importances_.reshape(1,-1), columns=grid.best_estimator_.named_steps[steps[1]].get_feature_names(), index=['Coefficients']).transpose()
        top_df.sort_values(by='Coefficients', inplace=True, ascending=False)
    return top_df
