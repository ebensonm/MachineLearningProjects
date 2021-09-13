import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import itertools
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score



class NaiveBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    '''

    def __init__(self):
        return 

    def fit(self, X, y):
        '''
        Create a table that will allow the filter to evaluate P(H), P(S)
        and P(w|C)

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        X = X.str.split()
        #get the list of unique words in X
        #create unique words list
        words_list = list(set(list(itertools.chain.from_iterable(X.tolist()))))
        #now create the new dataframe with appropriate indeces and columns
        self.data = pd.DataFrame(index=['ham','spam'], columns=words_list)
        #create the spam and ham mask
        y_ham = y == 'ham'
        y_spam = y == 'spam'
        #create the map from word counts to the column number
        X_ham = X[y_ham]
        X_ham = X_ham.apply(pd.Series).stack().tolist()
        X_spam = X[y_spam]
        X_spam = X_spam.apply(pd.Series).stack().tolist()
        for word in words_list:
            ham_count = X_ham.count(word)
            spam_count = X_spam.count(word)
            #now compute the counts for ham and spam
            self.data.loc['ham',word] = ham_count
            self.data.loc['spam',word] = spam_count
        self.words_list = words_list
        self.y_data = y

    def predict_proba(self, X):
        '''
        Find P(C=k|x) for each x in X and for each class k by computing
        P(C=k)P(x|C=k)

        Parameters:
            X (pd.Series)(N,): messages to classify
        
        Return:
            (ndarray)(N,2): Probability each message is ham, spam
                0 column is ham
                1 column is spam
        '''
        #define the apply function for computation of top part of equation 5
        def count_occurences(x):
            unique_x,counts = np.unique(np.array(x),return_counts=True)
            #get the total word counts
            total_counts = self.data.to_numpy(dtype='float').sum(axis=1)
            return_list = list()
            for count,word in zip(counts,unique_x):
                if word in self.words_list:
                    num = np.power(self.data[word].to_numpy(dtype='float')/total_counts,count)
                    return_list.append(num)
                else:
                    return_list.append(np.array([1.0,1.0],dtype='float'))
            return_array = np.product(np.stack(return_list),axis=0)
            return return_array
        #get shape and initialize matrices
        N = X.shape[0]
        #split the data into lists of words
        X = X.str.split()
        prob_array = np.zeros((N,2))
        #compute the total probabilities
        hams = np.sum(self.y_data == 'ham')
        spams = np.sum(self.y_data == 'spam')
        #compute the total ham probability
        total_ham_prob = hams/len(self.y_data)
        #compute the spam total probability
        total_spam_prob = spams/len(self.y_data)
        
        #compute the ham and spam conditionals
        conditionals = X.apply(count_occurences).to_list()
        conditionals = np.stack(conditionals)
        #multiply them together and add to the result
        probabilities = conditionals*np.array([total_ham_prob,total_spam_prob])
        prob_array = probabilities
        return prob_array

    def predict(self, X):
        '''
        Use self.predict_proba to assign labels to X,
        the label will be a string that is either 'spam' or 'ham'

        Parameters:
            X (pd.Series)(N,): messages to classify
        
        Return:
            (ndarray)(N,): label for each message
        '''
        #mapping function
        def number_to_pred(x):
            if (x == 0):
                return "ham"
            return "spam"
        #get the results
        prob_array = self.predict_proba(X)
        predictions = np.argmax(prob_array,axis=1)
        predictions = np.array(list(map(number_to_pred, predictions)),dtype='object')
        return predictions      
        
    def predict_log_proba(self, X):
        '''
        Find ln(P(C=k|x)) for each x in X and for each class k

        Parameters:
            X (pd.Series)(N,): messages to classify
        
        Return:
            (ndarray)(N,2): Probability each message is ham, spam
                0 column is ham
                1 column is spam
        '''
        #define the apply function for computation of top part of equation 5
        def count_occurences(x):
            unique_x,counts = np.unique(np.array(x),return_counts=True)
            #get the total word counts
            total_counts = self.data.to_numpy(dtype='float').sum(axis=1)
            return_list = list()
            for count,word in zip(counts,unique_x):
                if word in self.words_list:
                    num = count*np.log(self.data[word].to_numpy(dtype='float')/total_counts)
                    return_list.append(num)
                else:
                    return_list.append(np.array([1.0,1.0],dtype='float'))
            return_array = np.sum(np.stack(return_list),axis=0)
            return return_array
        #get shape and initialize matrices
        N = X.shape[0]
        #split the data into lists of words
        X = X.str.split()
        prob_array = np.zeros((N,2))
        #compute the total probabilities
        hams = np.sum(self.y_data == 'ham')
        spams = np.sum(self.y_data == 'spam')
        #compute the total ham probability
        total_ham_prob = hams/len(self.y_data)
        #compute the spam total probability
        total_spam_prob = spams/len(self.y_data)
        
        #compute the ham and spam conditionals
        conditionals = X.apply(count_occurences).to_list()
        conditionals = np.stack(conditionals)
        #multiply them together and add to the result
        probabilities = conditionals+np.log(np.array([total_ham_prob,total_spam_prob]))
        prob_array = probabilities
        return prob_array
        

    def predict_log(self, X):
        '''
        Use self.predict_log_proba to assign labels to X,
        the label will be a string that is either 'spam' or 'ham'

        Parameters:
            X (pd.Series)(N,): messages to classify
        
        Return:
            (ndarray)(N,): label for each message
        '''
        #mapping function
        def number_to_pred(x):
            if (x == 0):
                return "ham"
            return "spam"
        #get the results
        prob_array = self.predict_log_proba(X)
        predictions = np.argmax(prob_array,axis=1)
        predictions = np.array(list(map(number_to_pred, predictions)),dtype='object')
        return predictions

class PoissonBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    This classifier assumes that words are distributed like 
    Poisson random variables
    '''

    def __init__(self):
        return

    
    def fit(self, X, y):
        '''
        Uses bayesian inference to find the poisson rate for each word
        found in the training set. For this we will use the formulation
        of l = rt since we have variable message lengths.

        This method creates a tool that will allow the filter to 
        evaluate P(H), P(S), and P(w|C)


        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        
        Returns:
            self: this is an optional method to train
        '''
        #first format the data, and split the classes
        X = X.str.split()
        #get the list of unique words in X
        words_list = list(set(list(itertools.chain.from_iterable(X.tolist()))))
        #now create the new dataframe with appropriate indeces and columns
        #create the spam and ham mask
        y_ham = y == 'ham'
        y_spam = y == 'spam'
        data_spam = X[y_spam]
        data_ham = X[y_ham]
        self.spam_rates = dict()
        self.ham_rates = dict()
        #loop to do the same calculations on the data (for spam and ham)
        for keyword in ['ham','spam']:
            data = X[y==keyword].to_list()
            data = [word for words in data for word in words]
            #count the total number of words
            N_k = len(data)
            #compute n_i array
            n_i = np.array([data.count(word) for word in words_list])
            #do the optimization/minimization problem
            r = (n_i/N_k).tolist()
            if (keyword == 'ham'):
                self.ham_rates = {word:r_val for word,r_val in zip(words_list,r)}
            else:
                self.spam_rates = {word:r_val for word,r_val in zip(words_list,r)}
        self.y_data = y
        self.words_list = words_list
    
    def predict_proba(self, X):
        '''
        Find P(C=k|x) for each x in X and for each class

        Parameters:
            X (pd.Series)(N,): messages to classify
        
        Return:
            (ndarray)(N,2): Probability each message is ham or spam
                column 0 is ham, column 1 is spam 
        '''
        #define the function to be used to apply on the X's
        def compute_conditionals(x):
            #get the unique words and counts in the given array
            unique_x,counts = np.unique(np.array(x),return_counts=True)
            n = len(x)
            return_list = list()
            for count,word in zip(counts,unique_x):
                if word in self.words_list:
                    ham_rate = self.ham_rates[word]
                    spam_rate = self.spam_rates[word]
                    rates = np.array([ham_rate, spam_rate])
                    n_i = count
                    probability_num = np.log(np.power(n*rates,n_i)*np.exp(-(rates*n))/np.math.factorial(n_i))
                    return_list.append(probability_num)
                else:
                    return_list.append(np.array([0.0,0.0],dtype='float'))

            return_array = np.sum(np.stack(return_list),axis=0)
            return return_array
            
        #compute the total probabilities
        hams = np.sum(self.y_data == 'ham')
        spams = np.sum(self.y_data == 'spam')
        #compute the total ham probability
        total_ham_prob = hams/len(self.y_data)
        #compute the spam total probability
        total_spam_prob = spams/len(self.y_data)
        totals = np.array([total_ham_prob, total_spam_prob])
        
        #now compute the conditional probabilities from the new data
        data = X.str.split()
        #compute the ham and spam conditionals
        conditionals = data.apply(compute_conditionals).to_list()
        conditionals = np.stack(conditionals)
        #multiply them together and add to the result
        probabilities = conditionals+np.log(totals)
        prob_array = probabilities
        return prob_array
            
                

    def predict(self, X):
        '''
        Use self.predict_proba to assign labels to X

        Parameters:
            X (pd.Series)(N,): messages to classify
        
        Return:
            (ndarray)(N,): label for each message
        '''
        #mapping function
        def number_to_pred(x):
            if (x == 0):
                return "ham"
            return "spam"
        #get the results
        prob_array = self.predict_proba(X)
        predictions = np.argmax(prob_array,axis=1)
        predictions = np.array(list(map(number_to_pred, predictions)),dtype='object')
        return predictions 



def sklearn_method(X_train, y_train, X_test):
    '''
    Use sklearn's methods to transform X_train and X_test, create a
    naïve Bayes filter, and classify the provided test set.

    Parameters:
        X_train (pandas.Series): messages to train on
        y_train (pandas.Series): labels for X_train
        X_test  (pandas.Series): messages to classify

    Returns:
        (ndarray): classification of X_test
    '''
    #fit and transform the given data using count vectorizer
    word_handler = CountVectorizer()
    X_fit = word_handler.fit_transform(X_train,y_train)
    X_test_transform = word_handler.transform(X_test)
    #now train the NB classifier
    model = MultinomialNB()
    model.fit(X_fit,y_train)
    #get the prediction
    y_pred = model.predict(X_test_transform)
    return y_pred
    
