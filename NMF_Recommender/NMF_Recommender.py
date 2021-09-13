import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error

class NMFRecommender:

    def __init__(self,random_state=15,tol=1e-3,maxiter=200,rank=3):
        """The parameter values for the algorithm"""
        #set the class parameters
        self.random_state = random_state  
        self.tol = tol
        self.maxiter = maxiter
        self.rank = rank
        
    def initialize_matrices(self,m,n):
        """Initialize the W and H matrices"""
        #initialize the matrices randomly from random_state
        np.random.seed(self.random_state)
        W = np.random.uniform(size=(m,self.rank))
        H = np.random.uniform(size=(self.rank,n))
        return W,H
        
    def compute_loss(self,V,W,H):
        """Computes the loss of the algorithm according to the frobenius norm"""
        #compute the value of frobenius norm loss
        return np.linalg.norm(V-(W @ H), ord='fro')
    
    def update_matrices(self,V,W,H):
        """The multiplicative update step to update W and H"""
        #update the matrices using the given update steps
        H_new = H*(W.T@ V)/(W.T@ W@ H)
        W_new = W*(V@ H_new.T)/(W@ H_new@ H_new.T)
        return W_new, H_new
        
    def fit(self,V):
        """Fits W and H weight matrices according to the multiplicative update 
        algorithm. Return W and H"""
        m,n = np.shape(V)
        #initialize the matrices
        W, H = self.initialize_matrices(m,n)
        #begin the loop for computation
        for i in range(self.maxiter):
            W, H = self.update_matrices(V,W,H)
            loss = self.compute_loss(V,W,H)
            #stopping condition
            if (loss < self.tol):
                return W,H
        return W,H 

    def reconstruct(self, W, H):
        """Reconstructs the V matrix for comparison against the original V 
        matrix"""
        return W@ H
        
def prob4():
    """Run NMF recommender on the grocery store example"""
    V = np.array([[0,1,0,1,2,2],
                  [2,3,1,1,2,2],
                  [1,1,1,0,1,1],
                  [0,2,3,4,1,1],
                  [0,0,0,0,1,0]])
    #create and run the nmf recommender
    nmf = NMFRecommender(rank=2)
    W,H = nmf.fit(V)
    V_recon = nmf.reconstruct(W,H)
    #get the mask to check if component 2 is higher than component 1
    mask_h = H[0,:] < H[1,:]
    return W,H, np.sum(mask_h)
    
def prob5():
    """Calculate the rank and run NMF
    """
    #load the data
    data = pd.read_csv('artist_user.csv', index_col=0)
    index_col = data.index
    columns = data.columns
    #calculate the norm and benchmark value
    benchmark = np.linalg.norm(data, ord='fro') * 0.0001
    #initialize the rank value
    rank_val = 3
    #initialize RMSE
    RMSE = 100 * benchmark
    while (RMSE > benchmark):
        model = NMF(n_components=rank_val, init='random', random_state=0)
        #fit the model
        W = model.fit_transform(data)
        H = model.components_
        #compute V
        V = W@ H
        #compute the RMSE
        RMSE = np.sqrt(mean_squared_error(data, V))
        rank_val += 1
    V = pd.DataFrame(data=V,columns=columns,index=index_col)
    return rank_val-1,V
        
def discover_weekly(user_id, V):
    """
    Create the recommended weekly 30 list for a given user
    """
    data = pd.read_csv('artist_user.csv', index_col=0)
    artists = pd.read_csv('artists.csv', index_col=0)
    #get the user views
    users_orig = data.loc[user_id]
    users_views = V.loc[user_id]
    zero_mask = users_orig == 0
    non_zeros = users_orig[zero_mask]
    users_views = users_views[zero_mask]
    #now sort the series
    users_views = users_views.sort_values(ascending=False)
    #get the indeces
    users_index = list(users_views.index)
    users_index = [int(i) for i in users_index]
    #now select the parts using the given indeces
    artists_rec = artists.loc[users_index[:30]]
    return [[i] for i in list(artists_rec['name'])]
 
