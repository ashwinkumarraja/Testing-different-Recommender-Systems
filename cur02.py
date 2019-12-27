'''
CUR Algorithm:
Sparse matrix A can be represented as C*U*R where C is a matrix consisting of columns of A and R is a matrix consisting of rows of A.
C and R are sparse matrices while U is dense.
'''
import numpy as np
import pickle
from collections import Counter
from numpy.linalg import svd
import numpy.linalg as linalg
import time
import random

import os


all_user=[]
all_movie=[]
all_rating=[]

fileDir=""
filename=""


fileDir = os.path.dirname(os.path.realpath('__file__'))
filename = os.path.join(fileDir, './ratings.dat')

file = open(filename,'r')
Ras = np.zeros((6041, 3953))

for l in file.readlines():
    line = l.strip().split('::')
    all_user.append(int(line[0]))
    all_movie.append(int(line[1]))
    all_rating.append(float(line[2]))

file.close()

for p in range(1000000):
    Ras[all_user[p], all_movie[p]] = float(all_rating[p])

matrix = Ras   
n_users = matrix.shape[0]
n_movies = matrix.shape[1]
print(matrix)
# matrix=np.array([[1,1,1,0,0],[3,3,3,0,0],[4,4,4,0,0],[5,5,5,0,0],[0,2,0,4,4],[0,0,0,5,5],[0,1,0,2,2]])

# n_users = matrix.shape[0]
# n_movies = matrix.shape[1]


def Svd(matrix):
    users_cnt = matrix.shape[0]
    movies_cnt = matrix.shape[1]

    svd_time_start = time()

    transposed = 0
    if(users_cnt < movies_cnt):
        transposed = 1
        matrix = matrix.T

    sign_flipped = dict()
    m1 = np.dot(matrix.T,matrix)
    eigenValues, eigenVectors = linalg.eig(m1)
    eigenValues = eigenValues.real
    eigenVectors = eigenVectors.real
    eigenValues = np.asarray(eigenValues,dtype = 'float64')
    eigenVectors = np.asarray(eigenVectors,dtype = 'float64')
    eigen_map = dict()
    for i in range(len(list(eigenValues))):
        eigenValues[i] = round(eigenValues[i], 4)
    for i in range(len(eigenValues)):
        if eigenValues[i] != 0.0:
            if eigenVectors[0][i] > 0.0:
                eigen_map[eigenValues[i]] = (eigenVectors[:, i])
                sign_flipped[eigenValues[i]] = 0
            else:
                eigen_map[eigenValues[i]] = (eigenVectors[:, i])*(-1)
                sign_flipped[eigenValues[i]] = 1
    eigenValues = sorted(list(eigen_map.keys()), reverse=True)

    V = np.zeros(shape=(len(matrix[0]),len(list(eigenValues))), dtype='float64')
    for i in range(len(eigenValues)):
        V[:,i] = eigen_map[eigenValues[i]]
    V = V[:,:len(eigenValues)]

    Sigma = np.diag([i**0.5 for i in eigenValues])

    U = np.zeros(shape=(len(matrix),len(list(eigenValues))), dtype='float64')
    for i in range(len(eigenValues)):
        if (sign_flipped[eigenValues[i]] == 1):
            U[:,i] = (np.dot(matrix,V[:,i]))*((-1)/(eigenValues[i]**0.5))
        else:
            U[:,i] = (np.dot(matrix,V[:,i]))*(1/(eigenValues[i]**0.5))

    if transposed == 0:
        return U, Sigma, V.T
    else:
        return V, Sigma.T, U.T



start_time = time.time()
'''
The mean rating of each user is calculated
'''
users_mean = matrix.sum(axis=1)
counts = Counter(matrix.nonzero()[0])
for i in range(n_users):
    if i in counts.keys():
        users_mean[i] = users_mean[i]/counts[i]
    else:
        users_mean[i] = 0

'''
The mean rating of each movie is calculated
'''
movies_mean = matrix.T.sum(axis=1)
counts = Counter(matrix.T.nonzero()[0])
for i in range(n_movies):
    if i in counts.keys():
        movies_mean[i] = movies_mean[i]/counts[i]
    else:
        movies_mean[i] = 0

'''
The probabilities of selection along the columns and rows is calculated
'''
total_norm =  np.linalg.norm(matrix)
col_norm =  np.linalg.norm(matrix,axis = 0)
row_norm =  np.linalg.norm(matrix,axis = 1)
for i in range(n_movies):
    col_norm[i] = (col_norm[i]/total_norm)**2
    
for i in range(n_users):
    row_norm[i] = (row_norm[i]/total_norm)**2

'''
Using the probabilities calculated above, columns and rows are randomly selected from the sparse matrix
'''
c=3000
selected_col = []
C = np.zeros([n_users,c])
for i in range(c):
    selected_col.append(random.randint(0,n_movies-1))   #Columns selected randomly
i=0
duplicate = len(selected_col) - len(set(selected_col))
for x in selected_col:
    p = col_norm[x]
    d = np.sqrt(c*p)
    if duplicate == 0 and p!=0:
        C[:,i] = matrix[:,x]/d
    elif p!=0 and d!=0:
        C[:,i] = (matrix[:,x]/d)*(duplicate)**0.5
    else:
        C[:i]=0
    i = i+1

'''
Using the probabilities calculated above, columns and rows are randomly selected from the sparse matrix
'''
r=3000
selected_row = []
R = np.zeros([n_movies,r])
for i in range(r):
    selected_row.append(random.randint(0,n_users-1))    #Rows selected randomly
i=0
duplicate = len(selected_row) - len(set(selected_row))
for x in selected_row:
    p = row_norm[x]
    d = np.sqrt(r*p)
    if duplicate == 0 and d!=0:
        R[:,i] = matrix.T[:,x]/d
    elif duplicate!=0 and d!=0:
        R[:,i] = (matrix.T[:,x]/d)*(duplicate)**0.5
    else:
        R[:i]=0
    i = i+1

'''
The matrix U is constructed from W by the Moore-Penrose pseudoinverse
This step involves using SVD to find U and V' of W.
W is calculated as the intersection of the selected rows and columns
'''
W = C[selected_row,:]
W1, W_cur, W2 = svd(W)
W_cur = np.diag(W_cur)
total_sum = 0
dimensions = W_cur.shape[0]
for i in range(dimensions):
    total_sum = total_sum + np.square(W_cur[i,i])   #Find square of sum of all diagonals
retained = total_sum
while dimensions > 0:
    retained = retained - np.square(W_cur[dimensions-1,dimensions-1])
    if retained/total_sum < 1:        #90% energy retention
        break
    else:
        W1 = W1[:,:-1:]
        W2 = W2[:-1,:]
        W_cur = W_cur[:,:-1]
        W_cur = W_cur[:-1,:]
        dimensions = dimensions - 1 
for i in range(W_cur.shape[0]):
    W_cur[i][i] = 1/W_cur[i][i]
U = np.dot(np.dot(W2.T, W_cur**2), W1.T)
cur_100 = np.dot(np.dot(C, U), R.T)

'''
All ratings estimated to be greater than 5 or less than 0 are rewritten
'''
for i in range(cur_100.shape[0]):
    for j in range(cur_100.shape[1]):
        if cur_100[i,j] > 5:
            cur_100[i,j] = 5
        elif cur_100[i,j] < 0:
            cur_100[i,j] = 0

end_time = time.time()

def RMSE(pred,value):
    N = pred.shape[0]
    M = pred.shape[1]
    cur_sum = np.sum(np.square(pred-value))
    return np.sqrt(cur_sum/(N*M))

    
def MAE(pred,value):
    N = pred.shape[0]
    M = pred.shape[1]
    cur_sum = np.sum(np.absolute(np.array(pred)-np.array(value)))
    return (cur_sum/(N*M))    
    


print("RMSE:",RMSE(cur_100, matrix))
print("MAE:",MAE(cur_100, matrix))

print("Total time taken : ",end_time - start_time , "seconds")