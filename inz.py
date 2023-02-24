import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import operator
import pickle
#import openpyxl as xl

eps = np.finfo(float).eps

def mae(observed_value, predicted_value):
    mae = (np.abs(np.subtract(predicted_value,observed_value)).sum()/len(observed_value))
    return mae

def mse(observed_value, predicted_value):
	mse = (np.subtract(predicted_value,observed_value) ** 2).sum()/len(predicted_value)
	return mse

def rmse(observed_value, predicted_value):
	rmse = np.sqrt((np.subtract(predicted_value,observed_value) ** 2).sum()/len(predicted_value))
	return rmse

def train_test_split(X, Y, test_size = 0.2, random_state = None):
	X_test = X.sample(frac = test_size, random_state = random_state)
	Y_test = Y[X_test.index]
	X_train = X.drop(X_test.index)
	Y_train = Y.drop(Y_test.index)
	return X_train, X_test, Y_train, Y_test

class DecisionTree:
    def __init__(self, min_num_sample = 3, max_depth = 0):
        self.depth = 0
        self.max_depth = max_depth
        self.min_num_sample = min_num_sample
        self.coefficient_of_variation = 10
        self.features = list
        self.num_features = int
        self.X_train = np.array
        self.Y_train = np.array
        self.train_size = int
	
    def build_tree(self, dataFrame, tree = None):
        
        if tree is None:
            tree = {}
            tree[feature] = {}
        
        feature, cutoff = self.best_split(dataFrame)
        if cutoff is None:
            return tree
        
        #Left
        new_dataFrame = self.split_rows(operator.le, dataFrame, feature, cutoff)
        cov = self.cov(new_dataFrame['target'])
        self.depth += 1
        
        if(self.coefficient_of_variation > cov or len(new_dataFrame) <= self.min_num_sample):
            tree[feature]['<=' + str(cutoff)] = new_dataFrame['target'].mean()
        else:
            if self.max_depth is not None and self.depth >= self.max_depth:
                tree[feature]['<=' + str(cutoff)] = new_dataFrame['target'].mean()
            else:
                tree[feature]['<=' + str(cutoff)] = self.build_tree(new_dataFrame)
        
        #Right        
        new_dataFrame = self.split_rows(operator.gt, dataFrame, feature, cutoff)
        cov = self.cov(new_dataFrame['target'])
        
        if(self.coefficient_of_variation > cov or len(new_dataFrame) <= self.min_num_sample):
            tree[feature]['>' + str(cutoff)] = new_dataFrame['target'].mean()
        else:
            if self.max_depth is not None and self.depth >= self.max_depth:
                tree[feature]['>' + str(cutoff)] = new_dataFrame['target'].mean()
            else:
                tree[feature]['>' + str(cutoff)] = self.build_tree(new_dataFrame)
        return tree
    
    def fit(self, X, Y):
        self.features = list(X.columns)
        self.num_features = X.shape[1]
        self.X_train = X
        self.Y_train = Y
        self.train_size = X.shape[0]
        
        dataFrame = X.copy()
        dataFrame['target'] = Y.copy()
        
        self.tree = self.build_tree(dataFrame)
        #print("\nDecision Tree(depth = {}) : \n {}".format(self.depth, self.tree))
    
    def cov(self, Y):
        if(Y.std() == 0):
            return 0
        return (100 * (Y.mean()/Y.std()))
    
    def split_rows(self, operation, dataFrame, feature, feature_value):
        return dataFrame[operation(dataFrame[feature], feature_value)].reset_index(drop = True)
    
    def best_split(self, dataFrame):
        best_score = float('inf')
        best_feature = str
        cutoff = None
        for feature in list(dataFrame.columns[:-1]):
            treshold, score = self.feature_split(dataFrame, feature)
            if score < best_score:
                best_score = score
                best_feature = feature
                cutoff = treshold
                #print(best_feature)
                #print(best_score)
        return best_feature, cutoff
    
    def feature_split(self, dataFrame, feature):
        best_score = float('inf')
        cutoff = float
        
        for value in dataFrame[feature]:
            data_right = dataFrame[feature][dataFrame[feature] > value]
            data_left = dataFrame[feature][dataFrame[feature] <= value]
            
            if(len(data_left) > 0 and len(data_right) > 0):
                score = self.find_score(data_right, data_left, dataFrame)
                if best_score > score:
                    best_score = score
                    cutoff = value
        return cutoff, best_score
    
    def find_score(self, rhs, lhs, dataFrame):
        Y = dataFrame['target']
        rhs_s = Y.iloc[rhs.index].std()
        lhs_s = Y.iloc[lhs.index].std()
        
        if(np.isnan(rhs_s)):
            lhs_s = 0
        if(np.isnan(lhs_s)):
            rhs_s = 0
        return rhs_s * rhs.sum() + lhs_s * lhs.sum()
    
    def predict_target(self, feature, tree, X):
        for i in tree.keys():
            value = X[i]
            if type(value) == str:
                tree = tree[i][value]
            else:
                cutoff = str(list(tree[i].keys())[0].split('<=')[1])
                if(value <= float(cutoff)):
                    tree = tree[i]['<='+cutoff]
                else:
                    tree = tree[i]['>'+cutoff]
            predict = str
            
            if type(tree) is dict:
                predict = self.predict_target(feature, tree, X)
            else:
                predict = tree
                return predict
        return predict
    
    def predict(self, X):
        result = []
        feature = {key: i for i, key in enumerate(list(X.columns))}
        
        for i in range(len(X)):
            result.append(self.predict_target(feature, self.tree, X.iloc[i]))
        
        return np.array(result)

class ErrorTree:
    def __init__(self, mae_train, mse_train, rmse_train, mae_test, mse_test, rmse_test):
        self.mae_train = mae_train
        self.mse_train = mse_train
        self.rmse_train = rmse_train
        self.mae_test = mae_test
        self.mse_test = mse_test
        self.rmse_test = rmse_test

'''
#Speed
data_speed = pd.read_excel("dane.xls", sheet_name="Speed") 
X, Y = data_speed.drop(data_speed.columns[-1], axis = 1), data_speed[data_speed.columns[-1]]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

tree_speed = DecisionTree(max_depth=100)
tree_speed.fit(X, Y)

mae_train = mae(Y_train, tree_speed.predict(X_train))
mse_train = mse(Y_train, tree_speed.predict(X_train))
rmse_train = rmse(Y_train, tree_speed.predict(X_train))
    
mae_test = mae(Y_test, tree_speed.predict(X_test))
mse_test = mse(Y_test, tree_speed.predict(X_test))
rmse_test = rmse(Y_test, tree_speed.predict(X_test)) 

error_speed = ErrorTree(mae_train, mse_train, rmse_train, mae_test, mse_test, rmse_test)

with open('model_speed.pkl','wb') as model_speed:
        pickle.dump(tree_speed,model_speed)
        pickle.dump(error_speed, model_speed)

#Shear Rate
     
data_shear_rate = pd.read_excel("dane.xls", sheet_name="Shear Rate") 
X, Y = data_shear_rate.drop(data_shear_rate.columns[-1], axis = 1), data_shear_rate[data_shear_rate.columns[-1]]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

tree_shear_rate = DecisionTree(max_depth=100)
tree_shear_rate.fit(X, Y)

mae_train = mae(Y_train, tree_shear_rate.predict(X_train))
mse_train = mse(Y_train, tree_shear_rate.predict(X_train))
rmse_train = rmse(Y_train, tree_shear_rate.predict(X_train))
    
mae_test = mae(Y_test, tree_shear_rate.predict(X_test))
mse_test = mse(Y_test, tree_shear_rate.predict(X_test))
rmse_test = rmse(Y_test, tree_shear_rate.predict(X_test)) 

error_shear_rate= ErrorTree(mae_train, mse_train, rmse_train, mae_test, mse_test, rmse_test)

with open('model_shear_rate.pkl','wb') as model_shear_rate:
        pickle.dump(tree_shear_rate,model_shear_rate)
        pickle.dump(error_shear_rate, model_shear_rate)
        
#Shear Stress

data_shear_stress = pd.read_excel("dane.xls", sheet_name="Shear Stress") 
X, Y = data_shear_stress.drop(data_shear_stress.columns[-1], axis = 1), data_shear_stress[data_shear_stress.columns[-1]]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

tree_shear_stress = DecisionTree(max_depth=100)
tree_shear_stress.fit(X, Y)

mae_train = mae(Y_train, tree_shear_stress.predict(X_train))
mse_train = mse(Y_train, tree_shear_stress.predict(X_train))
rmse_train = rmse(Y_train, tree_shear_stress.predict(X_train))
    
mae_test = mae(Y_test, tree_shear_stress.predict(X_test))
mse_test = mse(Y_test, tree_shear_stress.predict(X_test))
rmse_test = rmse(Y_test, tree_shear_stress.predict(X_test)) 

error_shear_stress= ErrorTree(mae_train, mse_train, rmse_train, mae_test, mse_test, rmse_test)

with open('model_shear_stress.pkl','wb') as model_shear_stress:
        pickle.dump(tree_shear_stress,model_shear_stress)
        pickle.dump(error_shear_stress, model_shear_stress)
        
#Torque

data_torque = pd.read_excel("dane.xls", sheet_name="Torque") 
X, Y = data_torque.drop(data_torque.columns[-1], axis = 1), data_torque[data_torque.columns[-1]]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

tree_torque = DecisionTree(max_depth=100)
tree_torque.fit(X, Y)

mae_train = mae(Y_train, tree_torque.predict(X_train))
mse_train = mse(Y_train, tree_torque.predict(X_train))
rmse_train = rmse(Y_train, tree_torque.predict(X_train))
    
mae_test = mae(Y_test, tree_torque.predict(X_test))
mse_test = mse(Y_test, tree_torque.predict(X_test))
rmse_test = rmse(Y_test, tree_torque.predict(X_test)) 

error_torque= ErrorTree(mae_train, mse_train, rmse_train, mae_test, mse_test, rmse_test)

with open('model_torque.pkl','wb') as model_torque:
        pickle.dump(tree_torque,model_torque)
        pickle.dump(error_torque, model_torque)'''