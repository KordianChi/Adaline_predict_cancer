import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
%matplotlib inline

#Neuron implementation

class Adaline:
    
    def __init__(self, eta=0.1, epoch=50, is_verbose=False):
        
        self.eta = eta
        self.epoch = epoch
        self.is_verbose = is_verbose
        self.list_of_errors = []
        
    def predict(self, x):
        
        ones = np.ones((x.shape[0],1))
        x_1 = np.append(x.copy(), ones, axis=1)
        
        return np.where(self.get_activation(x_1) > 0, 1, -1)
    
    def get_activation(self, x):
        
        activation = np.dot(x, self.w)
        return activation
    
    def fit(self, X, y):
        
        self.list_of_errors = []
        
        ones = np.ones((X.shape[0],1))
        X_1 = np.append(X.copy(), ones, axis=1)
        
        self.w = np.random.rand(X_1.shape[1])
        
        for e in range(self.epoch):
            error = 0
            
            activation = self.get_activation(X_1)
            delta_w = self.eta * np.dot((y-activation), X_1)
            self.w += delta_w
            
            error = np.square(y - activation).sum()/2.0
            
            self.list_of_errors.append(error)
            
            if self.is_verbose:
                print(f"Epoch: {e}, weights: {self.w}, error {error}")

#Data preparation

diag = pd.read_csv('./breast_cancer.csv')

X = diag[['area_mean', 'area_se', 'texture_mean',
          'concavity_worst', 'concavity_mean']]

y = diag[['diagnosis']].squeeze()

my_dict = {'M' : 1, 'B' : -1}

y = y.map(my_dict)

epochs = 100

 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2)


#Train and test

model = Adaline(eta=0.001, epoch=100)
model.fit(X_train, y_train)
plt.scatter(range(model.epoch), model.list_of_errors)
 
y_pred = model.predict(X_test)
 
good = y_test[y_test == y_pred].count()
total = y_test.count()
print(f'result: {100*good/total} %')