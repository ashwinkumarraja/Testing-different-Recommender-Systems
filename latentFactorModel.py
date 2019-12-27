import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate as cv
import matplotlib.pyplot as plt
import os
import pickle
import math
from time import time

perm = np.arange(1000000)

mu = 0    # global bias - average of all the train data labels
lamda = 0.1 # Regularisation weight
k = 0  # Dimension of the latent feature space
m=0
n = 0  # Number of users and items
no_epochs = 100  # Number of epochs
alpha = 0.01  # Learning rate
mul = 0.01   # multiplication factor

all_user=[] #Array to store all user numbers
all_movie=[] #Array to store all movie numbers
all_rating=[] #Array to store corresponding rating for user and movie

fileDir=""
filename=""

def test():
	'''
	This function is used for cross-validation and testing with remaining 30% of the dataset. This code can be modified to directly use optimized P,Q,bias_u,bias_m values which have been pickled
	'''
	timer=0 #Used to evaluate average time taken per query
	diff1=0
	diff2=0
	for p in range(700001,1000000,1):
		u=all_user[perm[p]]
		m=all_movie[perm[p]]
		start_time = time()
		rating = min(5,(mu + bias_u[u] + bias_m[m] + np.dot(P[:,u].T,Q[:,m]))) #Prediction of rating
		end_time = time()
		timer += (end_time-start_time)
		try:
			diff1 += (all_rating[perm[p]] - rating)**2 #sum of squared errors
			diff2 += abs(all_rating[perm[p]] - rating) #absolute sum of errors
		except IndexError:
			break
	
	print("Average time for prediction = "+str(timer/300000)+" seconds")
	mse = diff1/300000
	mae = diff2/300000
	print("Root Mean Square Error in predicting test rating is: ",str(math.sqrt(mse)))
	print("Mean Average Error in predicting test rating is: ",str(mae))
	print("\n")

def train():
	'''
	This function trains and finds the optimal values of matrices P, Q and vectors bias_u, bias_m
	These were randomly initialized initially
	'''
	global mu
	for val in np.nditer(R):
		mu+=val

	mu=mu/700000

	for epoch in range(no_epochs):
		if epoch % 25==0:
			print("--------------EPOCH "+str(epoch)+"---------------\n")
			test()

		for u, i in zip(users,movies):
			e = R[u, i] - (mu + bias_u[u] + bias_m[i] + np.dot(P[:,u].T,Q[:,i]))  # Calculating error for gradient
			bias_m[i] += alpha*(e - lamda*bias_m[i]) #Updating bias for movies
			bias_u[u] += alpha*(e - lamda*bias_u[u]) #Updating bias for user
			P[:,u] += alpha * ( e * Q[:,i] - lamda * P[:,u]) # Update latent user feature matrix
			Q[:,i] += alpha * ( e * P[:,u] - lamda * Q[:,i])  # Update latent movie feature matrix

		print(epoch)

	# DUMPING OPTIMAL VALUES 
	with open(r"C:\Users\sriha\Desktop\IR finale\ASS3\Achar"+str(k)+".pickle","wb") as f:
		pickle.dump(mu, f)
		pickle.dump(bias_u, f)
		pickle.dump(bias_m, f)
		pickle.dump(P,f)
		pickle.dump(Q,f)


	

if "__name__==__main__":
	fileDir = os.path.dirname(os.path.realpath('__file__'))
	filename = os.path.join(fileDir, 'ml-1m/ml-1m/ratings.dat')

	file = open(filename,'r')
	R = np.zeros((6041, 3953)) #Rating matrix

	#Reading the file and appending to arrays
	for l in file.readlines():
		line = l.strip().split('::')
		all_user.append(int(line[0]))
		all_movie.append(int(line[1]))
		all_rating.append(float(line[2]))

	file.close()

	# Using first 700000 randomly permuted values from input data
	for p in range(700000):
		R[all_user[perm[p]], all_movie[perm[p]]] = float(all_rating[perm[p]])

	m,n = R.shape

	#Only considering non-zero matrix 
	users,movies = R.nonzero()

	#Trying multiple number of latent features
	latentArray = [100,200,300,500]

	for k in latentArray:
		print("\n!!!!!!!!!!!!!!!! Trial with k value = "+str(k)+"!!!!!!!!!!!!!!!!\n\n\n")

		bias_u = np.zeros((6041,1))    ## user bias
		bias_m = np.zeros((3953,1))    ## item bias

		P =  alpha*np.random.randn(k,m) # Latent user feature matrix
		Q =  alpha*np.random.randn(k,n) # Latent movie feature matrix
	
		train() #Training with train rating data