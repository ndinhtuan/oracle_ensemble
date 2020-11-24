import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
import pickle
from imblearn.over_sampling import SMOTE

class RandomOracleModel(object):

    def __init__(self, base_learning='svm'):
        
        self.model1 = None
        self.model2 = None
        self.inst1 = None 
        self.inst2 = None

        if base_learning=='svm':
            self.model1 = SVC(C=1.0, kernel='rbf', gamma='auto')
            self.model2 = SVC(C=1.0, kernel='rbf', gamma='auto')
        if base_learning=='smoteboost':
            self.model1 = AdaBoostClassifier()
            self.model2 = AdaBoostClassifier()
    
    def __distance(self, x1, x2):

        a = np.array(x1)
        b = np.array(x2)
        return np.linalg.norm(a-b)

    def save(self, path_state_dict):
        
        model_state = {} 
        model_state["instance1"] = self.inst1 
        model_state["instance2"] = self.inst2
        model_state["model1"] = self.model1 
        model_state["model2"] = self.model2

        with open(path_state_dict, 'wb') as file:
            pickle.dump(model_state, file)
            file.close()

    def load(self, path_state_dict):
        
        with open(path_state_dict, 'rb') as file:
            model_state = pickle.load(file)
            self.inst1 = model_state["instance1"]
            self.inst2 = model_state["instance2"]
            self.model1 = model_state["model1"]
            self.model2 = model_state["model2"]

    def train(self, X, Y):
        
        len_train = len(X)
        i1 = np.random.randint(len_train)
        i2 = np.random.randint(len_train)

        while i2==i1:
            i2 = np.random.randint(len_train)

        self.inst1 = inst1 = X[i1]
        self.inst2 = inst2 = X[i2]

        X1 = [] 
        Y1 = []
        X2 = [] 
        Y2 = []

        for x, y in zip(X, Y):
            if self.__distance(x, inst1) < self.__distance(x, inst2):
                X1.append(x)
                Y1.append(y)
            else:
                X2.append(x) 
                Y2.append(y)

        X1 = np.array(X1)
        Y1 = np.array(Y1)
        X2 = np.array(X2)
        Y2 = np.array(Y2)
        
        print("Training Model 1")
        self.model1.fit(X1, Y1)
        print("Training Model 2")
        self.model2.fit(X2, Y2)

    def predict(self, x):

        assert self.model1 is not None and self.model2 is not None
        assert self.inst1 is not None and self.inst2 is not None

        if self.__distance(x, self.inst1) < self.__distance(x, self.inst2):
            return self.model1.predict([x])[0]
        else:
            return self.model2.predict([x])[0]

    def evaluate(self, x_test, y_test):

        assert self.model1 is not None and self.model2 is not None
        assert self.inst1 is not None and self.inst2 is not None

        preds = []

        for x in x_test:
            preds.append(self.predict(x)) 

        preds = np.array(preds)
        
        return classification_report(y_test, preds)

class SmoteAdaboost(object):

    def __init__(self):

        self.adaboost_model = AdaBoostClassifier(n_estimator=10, random_state=42)

    def fit(self, X, Y):

        smote = SMOTE(random_state=42)
        X_sm, Y_sm = smote.fit_sample(X, Y)
        self.adaboost_model.fit(X_sm, Y_sm)

    def predict(self, x):
        return self.adaboost_model.predict(x)

# Show dataset and split dataset for training and testing
def data_analysis():

    df = pd.read_csv("emails.csv")
    len_neg = len(df[df["Prediction"]==0])
    len_pos = len(df[df["Prediction"]==1])

    print("{} positive vs {} negative".format(len_pos, len_neg))

    X = df.iloc[:, 1:3001]
    Y = df.iloc[:,-1].values
    
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.25)
    return train_x.to_numpy(), test_x.to_numpy(), \
            train_y, test_y

def test_svm():

    train_x, test_x, train_y, test_y = data_analysis()
    
    # Try with single classifier 
    svc = SVC(C=1.0, kernel='rbf', gamma='auto')
    svc.fit(train_x, train_y)
    y_pred = svc.predict(test_x)
    print("Accuracy Score for Single SVC: \n", classification_report(test_y, y_pred))
    
    # Try with Oracle Ensemble
    random_oracle_model = RandomOracleModel()
    random_oracle_model.train(train_x, train_y)
    print("Accuracy Score for Oracle ensemble: \n", random_oracle_model.evaluate(test_x, test_y))
    
    # Try Save and Load module
    random_oracle_model.save("model.pkl")
    rom = RandomOracleModel()
    rom.load("model.pkl")
    print("Accuracy Score for Saving and Loading of Oracle ensemble: \n", rom.evaluate(test_x, test_y))

def test_smoteadaboost():
    
    train_x, test_x, train_y, test_y = data_analysis()

    # Try with single classifier
    smoteada = AdaBoostClassifier()
    smoteada.fit(train_x, train_y)
    y_pred = smoteada.predict(test_x)
    print("Accuracy Score for Single SVC: \n", classification_report(test_y, y_pred))

    # Try with Oracle Ensemble
    random_oracle_model = RandomOracleModel(base_learning="smoteboost")
    random_oracle_model.train(train_x, train_y)
    print("Accuracy Score for Oracle ensemble: \n", random_oracle_model.evaluate(test_x, test_y))

    # Try Save and Load module
    random_oracle_model.save("model.pkl")
    rom = RandomOracleModel(base_learning="smoteboost")
    rom.load("model.pkl")
    print("Accuracy Score for Saving and Loading of Oracle ensemble: \n", rom.evaluate(test_x, test_y))

if __name__=="__main__":
    test_svm()
