from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import pandas as pd 
import numpy as np
import random

class BoostedRandomForest :
    def __init__(self, T=50, sample_portion=0.6, depth_max=5, criterion='entropy', eps_ub=1, eps_lb=0, eps_exceed_limit=5, weight_update=True, boosting=True, debug_msg=False, verbose=False) :
        # Inputs 
        # Max number of trees to be trained
        self.T = T
        # Portion of sampled subet from training data
        self.sample_portion = sample_portion
        # Max depth of each tree
        self.depth_max = depth_max
        # Determine if tree weights are updated during training
        self.weight_update = weight_update
        # Criterion to train a random tree
        self.criterion = criterion
        # Upper bound of epsilon to stop training trees
        self.eps_ub = eps_ub
        # Lower bound of epsilon to stop training trees
        self.eps_lb = eps_lb
        # Number of times epsilon is allowed to exceed the boundaries
        self.eps_exceed_limit = eps_exceed_limit
        # Determine if boosting is applied during training
        self.boosting = boosting
        # Determine if debug messages are printed
        self.debug_msg = debug_msg
        # Enable verbose output of training process
        self.verbose = verbose
        ###################################################################
        
        # Class variables
        # Number of times epsilon exceeeded boundaries
        self.eps_exceed_cnt = 0
        # List of trained randome tree classifiers
        self.clfs = []
        # List of weights to trained trees
        self.alphas = []
        # features selected for each tree in forest
        self.feature_record = pd.DataFrame()
        # List of accuracies during training
        self.train_accs = []
        # List of error rates for all trees trained (incl. rejected trees)
        self.all_eps = []
        # List of tree weights for all trees trained (incl. rejected trees)
        self.all_alphas = []
        ###################################################################
    
    
    # Train boosted random forest
    def fit(self, X, y) :
        # Number of output classes
        n_class = len(np.unique(y))
        # Number of features
        m = X.shape[0]
        # Number of examples
        N = X.shape[1]
        feature_portion = (np.round(np.sqrt(N))/N) *3
        
        # Initialize training sample weights
        W = [1.0/m for i in range(0,m,1)]
        W = pd.DataFrame({'Weight':list(W)}, index=X.index)
        
        # Print debug messages
        if self.debug_msg :
            print("Weight Update:", self.weight_update)
            print("Tree Boosting in forming forest:", self.boosting)
            print("max depthmax:", self.depth_max)
            print("feature_sampling:", feature_portion)
            print("training_sample:", self.sample_portion)
            print("--------------------------")
            
        for i in range(1,self.T+1):
            if self.verbose :
                print("in loop ", i )
            # Prepare training sample subset (bagging)
            X_train_sample, X_test_sample, y_train_sample, y_test_sample = train_test_split(X, y, test_size=self.sample_portion, shuffle=True)
            # Sample feautres to be used for current tree
            selected_features =  [random.randint(0,N-1) for j in range(0,int(round(N*feature_portion)))]
            X_train_sample = X_train_sample.iloc[:,selected_features]
            X_test_sample = X_test_sample.iloc[:,selected_features]
            # Save selected features for the tree
            self.feature_record = self.feature_record.append(pd.DataFrame([selected_features]), ignore_index=True)    
                        
            # Prepare tree classifier
            clf = tree.DecisionTreeClassifier(criterion=self.criterion, max_depth=self.depth_max)
            # Weight of training samples
            w_ = W.loc[X_train_sample.index,"Weight"].tolist()

            # Train decision tree
            clf.fit(X=X_train_sample, y=y_train_sample,sample_weight=w_)
            # Make prediction
            pred = clf.predict(X.iloc[:,selected_features])

            # Calculate weighted error rate of current tree
            eps = sum(np.array(W)[(np.ravel(pred) != np.ravel(y))]) / sum(np.array(W))
            if self.debug_msg :
                print("eps: ", eps)
                
            # Compute weight of decision tree
            alpha = (0.5)*np.log( (n_class-1)*(1-eps)/eps )
            if self.debug_msg:
                print("Alpha:", alpha)
                
            # Record eps and alpha
            self.all_eps.append(eps)
            self.all_alphas.append(alpha)
                
            # Stop training if the error rate is too small or too large
            if eps < self.eps_lb or self.eps_ub < eps :
                # Increment epsilon exceed counter
                self.eps_exceed_cnt += 1
                # If epsilon is outside limits too many times, 
                # stop training new trees.
                if self.eps_exceed_cnt > self.eps_exceed_limit :
                    if self.debug_msg :
                        print("eps == {}. Break".format(eps))
                    break
                

            # Update weight of training sample
            if alpha > 0 :
                # Calculate alphas according to correctness of predictions
                exp_alphas = [ np.exp(-alpha) if a==p else np.exp(alpha) for a,p in zip(np.ravel(y), pred) ]
                
                # Update training sample weights
                if self.weight_update==True:
                    #with updating
                    W = m*np.multiply(W, exp_alphas) / np.sum(np.multiply(W, exp_alphas))
                else:
                    #weihtout updating
                    W = [1.0/m for i in range(0,m,1)]
                    W = pd.DataFrame({'Weight':list(W)}, index=X.index)

                # Save trained tree to list 
                self.clfs.append(clf)
                # Save alpha to list
                self.alphas.append(alpha)

                if self.boosting==True:
                    alphas_ = self.alphas/sum(self.alphas)
                    
                    # Predict with trees trained so far
                    pred = self.ensemble_predict(X)
                    # Record accuracy
                    self.train_accs.append(accuracy_score(y, pred))
                else :
                    # Predict with trees trained so far
                    pred = self.RF_predict(X)
                    # Record accuracy
                    self.train_accs.append(accuracy_score(y, pred))
                    

            else :
                # If alpha < 0, reject the tree
                if self.debug_msg :
                    print("Tree {} is rejected.".format(i))
    
    
    def ensemble_predict(self, X) :
        # Normalize alphas
        alphas_ = self.alphas / sum(self.alphas)
        
        # Calculate class probabilities
        prob_mat=np.empty([0, X.shape[0]])
        for i in range(0,len(self.clfs)):
            prob = self.clfs[i].predict_proba(X.iloc[:, list(self.feature_record.iloc[i,:])])[:,1]
            prob_mat = np.vstack((prob_mat,prob))
        prob_mat = np.transpose(prob_mat)
        ensemble_prob = np.matmul(prob_mat,np.array(alphas_))
        
        # Give predictions
        ensemble_pred = ensemble_prob
        ensemble_pred[ensemble_prob>=0.5] = 1
        ensemble_pred[ensemble_prob<0.5] = 0
        
        return ensemble_pred
    
        
    # Give predictions with random trees
    def RF_predict(self, X) :
        # Calculate class probabilities
        prob_mat=np.empty([0, X.shape[0]])
        for i in range(0,len(self.clfs)):
            prob = self.clfs[i].predict_proba(X.iloc[:,list(self.feature_record.iloc[i,:])])[:,1]
            prob_mat = np.vstack ((prob_mat,prob))
        prob_mat = np.transpose(prob_mat)
        
        if len(self.clfs)>1:
            ensemble_prob = np.mean(prob_mat,axis=1)
        else:
            ensemble_prob = prob_mat
            
        # Give predictions
        ensemble_pred= ensemble_prob
        ensemble_pred[ensemble_prob>=0.5]=1
        ensemble_pred[ensemble_prob<0.5]=0
        
        return ensemble_pred
    
    