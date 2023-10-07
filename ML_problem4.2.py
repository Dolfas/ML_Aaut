
# ML PROBLEM 4.2
#100123 100260

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LassoCV, Lasso
from sklearn import linear_model 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import *
import warnings
import os
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore")

# Loading the data from the files given
X_train = np.load(os.path.abspath('X_train_regression2.npy'))
y_train = np.load(os.path.abspath('y_train_regression2.npy'))
X_test  = np.load(os.path.abspath('X_test_regression2.npy'))


# Kmeans com x e Y
# Método que faz a divisão em clusters com base no centróide das mesmas e da próximidade de cada dado ao centróide. Pode não ser o mais adequado para representar estes dados, uma vez que os mesmos não estão necessáriamente separados no espaço de 5 dimensões. Apenas sabemos que têm modelos lineares aplicados diferentes.
from sklearn.cluster import KMeans
kmeans_xy = KMeans(n_clusters = 2, random_state = 0, n_init='auto')
X_Y_Train = np.hstack((X_train,y_train))
X_Y_train_norm = preprocessing.normalize(X_Y_Train) 
#we can choose to train the clustering model with normalized or non normalized data
kmeans_xy.fit(X_Y_Train)
#separating x and y train data into 2 data sets (one for each linear model), in accordance with the separation made by the clustering method
model1_indices_xy =  [ind for ind, x in enumerate((kmeans_xy.labels_).tolist()) if x == 1]
X_train_model0_xy = X_train.copy()
X_train_model1_xy = X_train.copy()
y_train_model0_xy = y_train.copy()
y_train_model1_xy = y_train.copy()
X_train_model0_xy = np.delete(X_train_model0_xy, model1_indices_xy, 0)
y_train_model0_xy = np.delete(y_train_model0_xy, model1_indices_xy, 0)
X_train_model1_xy = X_train[model1_indices_xy,:]
y_train_model1_xy = y_train[model1_indices_xy,:]


# Gausian mixture model - Clustering
# 
# 
# Assume que os dados foram gerados por um número finito de distribuições gausianas com parâmetros desconhecidos, o que se adequa a este caso.
# Este lida melhor com a variância nos dados, que será gerada pelos modelos lineares, enquanto kmeans assume que os clusters têm igual variância
# 

from sklearn.mixture import GaussianMixture
gm = GaussianMixture(n_components=2, random_state=0).fit(X_Y_Train)
pred = gm.predict(X_Y_Train)
model1_indices_xy_gm =  [ind for ind, x in enumerate(pred) if x == 1]

X_train_model0_xy_gm = X_train.copy()
X_train_model1_xy_gm = X_train.copy()
y_train_model0_xy_gm = y_train.copy()
y_train_model1_xy_gm = y_train.copy()

X_train_model0_xy_gm = np.delete(X_train_model0_xy_gm, model1_indices_xy_gm, 0)
y_train_model0_xy_gm = np.delete(y_train_model0_xy_gm, model1_indices_xy_gm, 0)

X_train_model1_xy_gm = X_train[model1_indices_xy_gm,:]
y_train_model1_xy_gm = y_train[model1_indices_xy_gm,:]

        
# Funções de regressão por ridge com cálculo de r^2 e mse

def Ridge_func(x, y, cv_value):
    ridge = linear_model.RidgeCV(alphas=np.arange(0.01,10,0.5), scoring="neg_mean_squared_error", cv=cv_value).fit(x,y)    
    cv_results = cross_validate(ridge, x, y,scoring='neg_mean_squared_error', cv=cv_value)
    neg_mean_squared_errors = cv_results['test_score']
    MSE = np.mean(neg_mean_squared_errors)

    return MSE



def RidgeCv_otimization(x,y, fit_intercept_bool, cv_value_max):
    RidgeOptimization_model_parameters = np.empty([0,4])
    #we are going to test every possible value for k-fold cross validation including k=15 which is equivalent to the efficient Leave One Out method

    for cv_value in np.arange(2,cv_value_max+1,1):
        RidgeOptimization_model = linear_model.RidgeCV(alphas=np.arange(0.01,10,0.1), fit_intercept=fit_intercept_bool, scoring="r2", cv=cv_value).fit(x, y)
        
        cv_results = cross_validate(RidgeOptimization_model, x, y,scoring='neg_mean_squared_error', cv=cv_value_max+1)

        neg_mean_squared_errors = cv_results['test_score']
        mean_neg_mean_squared_error = np.mean(neg_mean_squared_errors)

        y_pred = RidgeOptimization_model.predict(x)

        
        r_cv_squared = r2_score(y, y_pred)

        RidgeOptimization_model_parameters = np.vstack((RidgeOptimization_model_parameters, np.array([cv_value , RidgeOptimization_model.alpha_, mean_neg_mean_squared_error, r_cv_squared])))
    MSE_otimization_2(RidgeOptimization_model_parameters)
    return(RidgeOptimization_model_parameters)

def MSE_otimization_2(matrix_cv_alpha_score):
    i = np.argmax(matrix_cv_alpha_score[:,2])
    print("The values that optimize the mean squared error are  ")
    print("Ridge Alpha " + str(matrix_cv_alpha_score[i,1]))
    print("Cv for least MSE  " + str(matrix_cv_alpha_score[i,0]))
    print("For a MSE  " + str(matrix_cv_alpha_score[i,2]))
    #print("MSE calculated via ridgecv - with fewer coluns in x=" + str(matrix_cv_alpha_score[i,2]))
    print('R^2  ' + str(matrix_cv_alpha_score[i,3]))
     





# **Modelo de regressão linear para todos os dados**

print('Para todos os dados \n ')
Ridge_parameters_all = RidgeCv_otimization(X_train,y_train,True,14)

print('\n \nPara o modelo 0 - kmens \n')
Ridge_parameters_kmeans0 = RidgeCv_otimization(X_train_model0_xy,y_train_model0_xy,True,14)

print('\n \nPara o modelo 1 - kmens \n')
Ridge_parameters_kmeans1 = RidgeCv_otimization(X_train_model1_xy,y_train_model1_xy,True,14)

print('\n \nPara o modelo 0 GMM\n')
Ridge_parameters_gm0 = RidgeCv_otimization(X_train_model0_xy_gm,y_train_model0_xy_gm,True,14)

print('\n \nPara o modelo 1 GMM\n')
Ridge_parameters_gm1 = RidgeCv_otimization(X_train_model1_xy_gm,y_train_model1_xy_gm,True,14)


# **Resultado do codigo anterior guardado**
# Para todos os dados 
#  
# The values that optimize the mean squared error are:
# Ridge Alpha 9.51
# Cv for least MSE6.0
# For a MSE=-1.3572364288447052
# R^2  0.2982846273721469
# 
#  
# Para o modelo 0
# 
# The values that optimize the mean squared error are:
# Ridge Alpha 9.51
# Cv for least MSE10.0
# For a MSE=-0.6662498966955057
# R^2  0.2606623865286021
# 
#  
# Para o modelo 1
# 
# The values that optimize the mean squared error are:
# Ridge Alpha 9.51
# Cv for least MSE2.0
# For a MSE=-0.5320787420658964
# R^2  0.19539112686604343
# 
#  
# Para o modelo 0 GMM
# 
# The values that optimize the mean squared error are:
# Ridge Alpha 0.01
# Cv for least MSE6.0
# For a MSE=-0.054203967258250975
# R^2  0.9715035608558902
# 
#  
# Para o modelo 1 GMM
# 
# The values that optimize the mean squared error are:
# Ridge Alpha 0.01
# Cv for least MSE3.0
# For a MSE=-0.03812143480853756
# R^2  0.980517166751606
# 
# Para o modelo 0 GMM normalizado
# 
# The values that optimize the mean squared error are  
# Ridge Alpha 0.01
# Cv for least MSE  7.0
# For a MSE  -0.05442421493082547
# R^2  0.9708523477639678
# 
#  
# Para o modelo 1 GMM normalizado
# 
# The values that optimize the mean squared error are  
# Ridge Alpha 0.01
# Cv for least MSE  4.0
# For a MSE  -0.03803255354735617
# R^2  0.9803674995642152

# Daqui concluímos que:
# - O melhor método é usar o Gausian para separar os dados que foram criados com cada um dos dois modelos
# - A partir destes dados, devemos usar um modelo ridge com alpha=0,01 para obter o melhor fit para os dados

def prediction_ridge(x,y, fit_intercept_bool, xtest):
    ridge = linear_model.RidgeCV(alphas=np.arange(0.01,10,0.1), fit_intercept=fit_intercept_bool, scoring="neg_mean_squared_error", cv=6).fit(x, y)
    ytest = ridge.predict(xtest)
    return ytest

# Entrega
y_test_model0_gm=prediction_ridge(X_train_model0_xy_gm,y_train_model0_xy_gm,True, X_test)
y_test_model1_gm=prediction_ridge(X_train_model1_xy_gm,y_train_model1_xy_gm,True, X_test)
prediction = np.hstack((y_test_model0_gm,y_test_model1_gm))
np.save('MLprediction_100123_100260_4.2',prediction)

# **Anexos**

# Comparison of the predictions given by Model 0 and Model 1
# Plotting the results
plt.figure(figsize=(18,10))
plt.plot(y_test_model0_gm[0:700],label='Model 0 prediction')
plt.plot(y_test_model1_gm[0:700],label='Model 1 prediction')
plt.title('Predicted Outcome: Model 0 vs Model 1')
plt.legend()




