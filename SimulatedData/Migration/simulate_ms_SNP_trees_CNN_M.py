#!/usr/bin/python3

## in order to use this code you have to have ms installed on your computer
## ms can be freely downloaded from:
## http://home.uchicago.edu/rhudson1/source/mksamples.html

import random
import os
import math
import shlex, subprocess
import numpy as np

##define a function to read ms' simulations and transform then into a NumPy array.    
def ms2nparray(xfile):
	g = list(xfile)
	k = [idx for idx,i in enumerate(g) if len(i) > 0 and i.startswith(b'//')]
	f = []
	for i in k:
		L = g[i+5:i+N_allpops+5]
		q = []
		for i in L:
			#originally: i = [int(j) for j in list(i)], need to decode as python3 loads as ASCII (0->48; 1->49).
			i = [int(j) for j in list(i.decode('utf-8'))]
			i = np.array(i, dtype=np.int8)
			q.append(i)
		q = np.array(q)
		q = q.astype("int8")
		f.append(np.array(q))
	return f

##define a function to read ms' coalescent trees simulations and add them to a list.
def get_newick(xfile):
	g = list(xfile)
	k = [idx for idx,i in enumerate(g) if len(i) > 0 and i.startswith(b'//')]
	t = []
	for i in k:
		n = g[i+1]
		t.append(n.decode('utf-8'))
	return t



def get_interspecific_tau(theta_a, theta_b, p_min=0.5, p_max=1.0, T_anc=None):
    """
    Ensures both populations are above the p_min threshold.
    The larger population (max theta) is the limiting factor.
    """
    limiting_theta = max(theta_a, theta_b)
    # We use a loop to ensure T is younger than T_parent.
    while True:
        p_target = np.random.uniform(p_min, p_max)
        # Calculate tau based on the lineage that takes longer to sort.
        ## obtain divergence time (tau - τ) priors using the P values. e.g., Pi_A = 1−2/3*e^(−2τ/θ_A); τ = -(θ_A*ln(-(3*Pi_A-3)/2)/2).
        tau = -((limiting_theta)*math.log(-(3*p_target-3)/2)/2)
        ## Transform divergence times to 4Ne generations units (required by ms).
        T = 4*tau/limiting_theta
        # If there's no parent, or if T is younger than parent return calculated values.
        if T_anc is None or T < T_anc:
            return T, p_target

def get_intraspecific_tau(theta_a, theta_b, p_min=1/3, p_max=0.4):
    """
    Ensures both populations are below the p_max threshold.
    The smaller population (min theta) is the limiting factor.
    """
    limiting_theta = min(theta_a, theta_b)
    p_target = np.random.uniform(p_min, p_max)
    
    # Calculate tau based on the lineage that sorts faster. 	
    ## obtain divergence time (tau - τ) priors using the P values. e.g., Pi_A = 1−2/3*e^(−2τ/θ_A); τ = -(θ_A*ln(-(3*Pi_A-3)/2)/2).
    tau = -((limiting_theta)*math.log(-(3*p_target-3)/2)/2)
    ## Transform divergence times to 4Ne generations units (required by ms).
    T = 4*tau/limiting_theta
    return abs(T),p_target
### variable declarations

#define the number of simulations
Priorsize = 10000

##Define sample sizes for each population. Multiply the number of individual by 2 for diploids.
## sample size of popA.
N_popA = 5*2
## sample size of popB.
N_popB = 5*2
## sample size of popC.
N_popC = 5*2
## sample size of popD.
N_popD = 5*2
## sample size of popE.
N_popE= 5*2
## sample size of popF.
N_popF = 5*2

## sample size for all pops combined.
N_allpops = N_popA + N_popB + N_popC + N_popD + N_popE + N_popF

## create files to store parameters, trees and the models
os.mkdir("trainingSims")
par_2sp_0125 = open("par_2sp_0125.txt","w")
par_3sp_0125 = open("par_3sp_0125.txt","w")
par_2sp_025 = open("par_2sp_025.txt","w")
par_3sp_025 = open("par_3sp_025.txt","w")
par_2sp_05 = open("par_2sp_05.txt","w")
par_3sp_05 = open("par_3sp_05.txt","w")
trees_2sp_0125 = []
trees_3sp_0125 = []
trees_2sp_025 = []
trees_3sp_025 = []
trees_2sp_05 = []
trees_3sp_05 = []
Model_2sp_0125 = []
Model_3sp_0125 = []
Model_2sp_025 = []
Model_3sp_025 = []
Model_2sp_05 = []
Model_3sp_05 = []

###Migration 0.125
## Two species, AB and CDEF
for i in range(Priorsize):

	## baseline Theta (4Neu) value of 0.005
	Theta = 0.005
	#Now sample relative Thetas (pop sizes) for every deme.
	Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F = np.random.uniform(0.5, 2, size=6)

	#And sample migration rates between Nm = 1 and 5 for all of the demes.
	pops = ['A','B','C','D','E','F']
	n = len(pops)

	# draw every possible rate (including i→i) between Nm = 1 and 5 (ms requires migration rates as 4Nm, so bounds are 4 and 20)
	M = np.random.uniform(4, 20, size=(n, n))
	# zero out self-migration
	np.fill_diagonal(M, 0)

	#Now change it for different species (AB,CDEF) with a value of m (Nm) = 0.125; 4Nm = 0.5
	for i in [0,1]:
		for j in [2,3,4,5]:
			M[i, j] = 0.5  # A→C, A→D, …, B→F
			M[j, i] = 0.5  # C→A, D→A, …, F→B
	
	## Sample Pi values for the deepest node (tau_1) between 0.5 and 1 (different species).
	T1, P1 = get_interspecific_tau(Theta*Theta_A, Theta*Theta_C)

	## Sample Pi values for the second deepest node (tau_2) between 1/3 (minimum value -> 0.333) and 0.4 (same species).
	T2, P2 = get_intraspecific_tau(Theta*Theta_C, Theta*Theta_E)

	## Sample the more recent splitting times with a smaller value than the more ancient ones and bigger than 0 (present)
	T3 = np.random.uniform(0, T2)
	T4 = np.random.uniform(0, T3)
	T5 = np.random.uniform(0, T4)
	## ms command
	com=subprocess.Popen("./ms %d 1000 -s 1 -t %f -I 6 %d %d %d %d %d %d -n 1 %f -n 2 %f -n 3 %f -n 4 %f -n 5 %f -n 6 %f -ma %s -ej %f 2 1 -ej %f 4 3 -ej %f 6 5 -ej %f 5 3 -ej %f 3 1 -T" % (N_allpops, Theta, N_popA, N_popB, N_popC, N_popD, N_popE, N_popF, Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F, ' '.join(map(str, M.flatten())), T5, T4, T3, T2, T1), shell=True, stdout=subprocess.PIPE).stdout

	output = com.read().splitlines()
	Model_2sp_0125.append(np.array(ms2nparray(output)).swapaxes(0,1).reshape(N_allpops,-1).T)

	## save parameter values and trees
	par_2sp_0125.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (
        Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F,
        T1, T2, T3, T4, T5))
	#Randomly save a number of tree equivalent to the number of traits
	trees_2sp_0125.append(random.sample(get_newick(output),100))
	print("Completed %d %% of Model 2sp_0125 simulations" % (float(i)/Priorsize*100))

#Save the simulated SNP data
Model_2sp_0125=np.array(Model_2sp_0125)
np.savez_compressed('trainingSims/Model_2sp_0125.npz', Model_2sp_0125=Model_2sp_0125)
del(Model_2sp_0125)


## Three species
for i in range(Priorsize):

	### Define parameters
	## baseline Theta (4Neu) value of 0.005
	Theta = 0.005
	#Now sample relative Thetas (pop sizes) for every deme.
	Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F = np.random.uniform(0.5, 2, size=6)

	#And sample migration rates between Nm = 1 and 5 for all of the demes.
	pops = ['A','B','C','D','E','F']
	n = len(pops)

	# draw every possible rate (including i→i) between Nm = 1 and 5 (ms requires migration rates as 4Nm, so bounds are 4 and 20)
	M = np.random.uniform(4, 20, size=(n, n))
	# zero out self-migration
	np.fill_diagonal(M, 0)

	#Now change it for different species (AB,CD,EF) with a value of m (Nm) = 0.125; 4Nm = 0.5
	species = [
    [0, 1],  # A,B
    [2, 3],  # C,D
    [4, 5],  # E,F
	]
	
	for sp in species:
		for i in sp:
			for j in range(n):
				if j not in sp:
					M[i, j] = 0.5
					M[j, i] = 0.5
	
	## Sample Pi values for the deepest node (tau_1) between 0.5 and 1 (different species).
	T1, P1 = get_interspecific_tau(Theta*Theta_A, Theta*Theta_C)

	## Sample Pi values for the second deepest node (tau_2) higher than 0.5 (different species), but lower than tau_1.
	T2, P2 = get_interspecific_tau(Theta*Theta_C, Theta*Theta_E, p_max=P1, T_anc=T1)

	## Sample Pi values for the third deepest node (tau_3) between 1/3 (minimum value -> 0.333) and 0.4 (same species).
	T3, P3 = get_intraspecific_tau(Theta*Theta_E, Theta*Theta_F)

	## Sample the more recent splitting times with a smaller value than the more ancient ones and bigger than 0 (present)
	T4 = np.random.uniform(0, T3)
	T5 = np.random.uniform(0, T4)

	## ms command
	com=subprocess.Popen("./ms %d 1000 -s 1 -t %f -I 6 %d %d %d %d %d %d -n 1 %f -n 2 %f -n 3 %f -n 4 %f -n 5 %f -n 6 %f -ma %s -ej %f 2 1 -ej %f 4 3 -ej %f 6 5 -ej %f 5 3 -ej %f 3 1 -T" % (N_allpops, Theta, N_popA, N_popB, N_popC, N_popD, N_popE, N_popF, Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F, ' '.join(map(str, M.flatten())), T5, T4, T3, T2, T1), shell=True, stdout=subprocess.PIPE).stdout

	output = com.read().splitlines()
	Model_3sp_0125.append(np.array(ms2nparray(output)).swapaxes(0,1).reshape(N_allpops,-1).T)
	
	## save parameter values and trees
	par_3sp_0125.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (
        Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F,
        T1, T2, T3, T4, T5))
	#Randomly save a number of tree equivalent to the number of traits
	trees_3sp_0125.append(random.sample(get_newick(output),100))
	print("Completed %d %% of Model 3sp_0125 simulations" % (float(i)/Priorsize*100))


#Save the simulated SNP data
Model_3sp_0125=np.array(Model_3sp_0125)
np.savez_compressed('trainingSims/Model_3sp_0125.npz', Model_3sp_0125=Model_3sp_0125)
del(Model_3sp_0125)

###Migration 0.25
## Two species, AB and CDEF
for i in range(Priorsize):

	## baseline Theta (4Neu) value of 0.005
	Theta = 0.005
	#Now sample relative Thetas (pop sizes) for every deme.
	Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F = np.random.uniform(0.5, 2, size=6)

	#And sample migration rates between Nm = 1 and 5 for all of the demes.
	pops = ['A','B','C','D','E','F']
	n = len(pops)

	# draw every possible rate (including i→i) between Nm = 1 and 5 (ms requires migration rates as 4Nm, so bounds are 4 and 20)
	M = np.random.uniform(4, 20, size=(n, n))
	# zero out self-migration
	np.fill_diagonal(M, 0)

	#Now change it for different species (AB,CDEF) with a value of m (Nm) = 0.25; 4Nm = 1
	for i in [0,1]:
		for j in [2,3,4,5]:
			M[i, j] = 1  # A→C, A→D, …, B→F
			M[j, i] = 1  # C→A, D→A, …, F→B
	
	## Sample Pi values for the deepest node (tau_1) between 0.5 and 1 (different species).
	T1, P1 = get_interspecific_tau(Theta*Theta_A, Theta*Theta_C)

	## Sample Pi values for the second deepest node (tau_2) between 1/3 (minimum value -> 0.333) and 0.4 (same species).
	T2, P2 = get_intraspecific_tau(Theta*Theta_C, Theta*Theta_E)

	## Sample the more recent splitting times with a smaller value than the more ancient ones and bigger than 0 (present)
	T3 = np.random.uniform(0, T2)
	T4 = np.random.uniform(0, T3)
	T5 = np.random.uniform(0, T4)

	## ms command
	com=subprocess.Popen("./ms %d 1000 -s 1 -t %f -I 6 %d %d %d %d %d %d -n 1 %f -n 2 %f -n 3 %f -n 4 %f -n 5 %f -n 6 %f -ma %s -ej %f 2 1 -ej %f 4 3 -ej %f 6 5 -ej %f 5 3 -ej %f 3 1 -T" % (N_allpops, Theta, N_popA, N_popB, N_popC, N_popD, N_popE, N_popF, Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F, ' '.join(map(str, M.flatten())), T5, T4, T3, T2, T1), shell=True, stdout=subprocess.PIPE).stdout

	output = com.read().splitlines()
	Model_2sp_025.append(np.array(ms2nparray(output)).swapaxes(0,1).reshape(N_allpops,-1).T)

	## save parameter values and trees
	par_2sp_025.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (
        Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F,
        T1, T2, T3, T4, T5))
	#Randomly save a number of tree equivalent to the number of traits
	trees_2sp_025.append(random.sample(get_newick(output),100))
	print("Completed %d %% of Model 2sp_025 simulations" % (float(i)/Priorsize*100))

#Save the simulated SNP data
Model_2sp_025=np.array(Model_2sp_025)
np.savez_compressed('trainingSims/Model_2sp_025.npz', Model_2sp_025=Model_2sp_025)
del(Model_2sp_025)


## Three species
for i in range(Priorsize):

	### Define parameters
	## baseline Theta (4Neu) value of 0.005
	Theta = 0.005
	#Now sample relative Thetas (pop sizes) for every deme.
	Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F = np.random.uniform(0.5, 2, size=6)

	#And sample migration rates between Nm = 1 and 5 for all of the demes.
	pops = ['A','B','C','D','E','F']
	n = len(pops)

	# draw every possible rate (including i→i) between Nm = 1 and 5 (ms requires migration rates as 4Nm, so bounds are 4 and 20)
	M = np.random.uniform(4, 20, size=(n, n))
	# zero out self-migration
	np.fill_diagonal(M, 0)

	#Now change it for different species (AB,CD,EF) with a value of m (Nm) = 0.25; 4Nm = 1
	species = [
    [0, 1],  # A,B
    [2, 3],  # C,D
    [4, 5],  # E,F
	]
	
	for sp in species:
		for i in sp:
			for j in range(n):
				if j not in sp:
					M[i, j] = 1
					M[j, i] = 1
	
	## Sample Pi values for the deepest node (tau_1) between 0.5 and 1 (different species).
	T1, P1 = get_interspecific_tau(Theta*Theta_A, Theta*Theta_C)

	## Sample Pi values for the second deepest node (tau_2) higher than 0.5 (different species), but lower than tau_1.
	T2, P2 = get_interspecific_tau(Theta*Theta_C, Theta*Theta_E, p_max=P1, T_anc=T1)

	## Sample Pi values for the third deepest node (tau_3) between 1/3 (minimum value -> 0.333) and 0.4 (same species).
	T3, P3 = get_intraspecific_tau(Theta*Theta_E, Theta*Theta_F)

	## Sample the more recent splitting times with a smaller value than the more ancient ones and bigger than 0 (present)
	T4 = np.random.uniform(0, T3)
	T5 = np.random.uniform(0, T4)

	## ms command
	com=subprocess.Popen("./ms %d 1000 -s 1 -t %f -I 6 %d %d %d %d %d %d -n 1 %f -n 2 %f -n 3 %f -n 4 %f -n 5 %f -n 6 %f -ma %s -ej %f 2 1 -ej %f 4 3 -ej %f 6 5 -ej %f 5 3 -ej %f 3 1 -T" % (N_allpops, Theta, N_popA, N_popB, N_popC, N_popD, N_popE, N_popF, Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F, ' '.join(map(str, M.flatten())), T5, T4, T3, T2, T1), shell=True, stdout=subprocess.PIPE).stdout

	output = com.read().splitlines()
	Model_3sp_025.append(np.array(ms2nparray(output)).swapaxes(0,1).reshape(N_allpops,-1).T)
	
	## save parameter values and trees
	par_3sp_025.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (
        Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F,
        T1, T2, T3, T4, T5))
	#Randomly save a number of tree equivalent to the number of traits
	trees_3sp_025.append(random.sample(get_newick(output),100))
	print("Completed %d %% of Model 3sp_025 simulations" % (float(i)/Priorsize*100))

#Save the simulated SNP data
Model_3sp_025=np.array(Model_3sp_025)
np.savez_compressed('trainingSims/Model_3sp_025.npz', Model_3sp_025=Model_3sp_025)
del(Model_3sp_025)

###Migration 0.5
## Two species, AB and CDEF
for i in range(Priorsize):

	## baseline Theta (4Neu) value of 0.005
	Theta = 0.005
	#Now sample relative Thetas (pop sizes) for every deme.
	Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F = np.random.uniform(0.5, 2, size=6)

	#And sample migration rates between Nm = 1 and 5 for all of the demes.
	pops = ['A','B','C','D','E','F']
	n = len(pops)

	# draw every possible rate (including i→i) between Nm = 1 and 5 (ms requires migration rates as 4Nm, so bounds are 4 and 20)
	M = np.random.uniform(4, 20, size=(n, n))
	# zero out self-migration
	np.fill_diagonal(M, 0)

	#Now change it for different species (AB,CDEF) with a value of m (Nm) = 0.5; 4Nm = 2
	for i in [0,1]:
		for j in [2,3,4,5]:
			M[i, j] = 2  # A→C, A→D, …, B→F
			M[j, i] = 2  # C→A, D→A, …, F→B
	
	## Sample Pi values for the deepest node (tau_1) between 0.5 and 1 (different species).
	T1, P1 = get_interspecific_tau(Theta*Theta_A, Theta*Theta_C)

	## Sample Pi values for the second deepest node (tau_2) between 1/3 (minimum value -> 0.333) and 0.4 (same species).
	T2, P2 = get_intraspecific_tau(Theta*Theta_C, Theta*Theta_E)

	## Sample the more recent splitting times with a smaller value than the more ancient ones and bigger than 0 (present)
	T3 = np.random.uniform(0, T2)
	T4 = np.random.uniform(0, T3)
	T5 = np.random.uniform(0, T4)

	## ms command
	com=subprocess.Popen("./ms %d 1000 -s 1 -t %f -I 6 %d %d %d %d %d %d -n 1 %f -n 2 %f -n 3 %f -n 4 %f -n 5 %f -n 6 %f -ma %s -ej %f 2 1 -ej %f 4 3 -ej %f 6 5 -ej %f 5 3 -ej %f 3 1 -T" % (N_allpops, Theta, N_popA, N_popB, N_popC, N_popD, N_popE, N_popF, Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F, ' '.join(map(str, M.flatten())), T5, T4, T3, T2, T1), shell=True, stdout=subprocess.PIPE).stdout

	output = com.read().splitlines()
	Model_2sp_05.append(np.array(ms2nparray(output)).swapaxes(0,1).reshape(N_allpops,-1).T)

	## save parameter values and trees
	par_2sp_05.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (
        Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F,
        T1, T2, T3, T4, T5))
	#Randomly save a number of tree equivalent to the number of traits
	trees_2sp_05.append(random.sample(get_newick(output),100))
	print("Completed %d %% of Model 2sp_05 simulations" % (float(i)/Priorsize*100))

#Save the simulated SNP data
Model_2sp_05=np.array(Model_2sp_05)
np.savez_compressed('trainingSims/Model_2sp_05.npz', Model_2sp_05=Model_2sp_05)
del(Model_2sp_05)


## Three species
for i in range(Priorsize):

	### Define parameters
	## baseline Theta (4Neu) value of 0.005
	Theta = 0.005
	#Now sample relative Thetas (pop sizes) for every deme.
	Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F = np.random.uniform(0.5, 2, size=6)

	#And sample migration rates between Nm = 1 and 5 for all of the demes.
	pops = ['A','B','C','D','E','F']
	n = len(pops)

	# draw every possible rate (including i→i) between Nm = 1 and 5 (ms requires migration rates as 4Nm, so bounds are 4 and 20)
	M = np.random.uniform(4, 20, size=(n, n))
	# zero out self-migration
	np.fill_diagonal(M, 0)

	#Now change it for different species (AB,CD,EF) with a value of m (Nm) = 0.5; 4Nm = 2
	species = [
    [0, 1],  # A,B
    [2, 3],  # C,D
    [4, 5],  # E,F
	]
	
	for sp in species:
		for i in sp:
			for j in range(n):
				if j not in sp:
					M[i, j] = 2
					M[j, i] = 2
	
	## Sample Pi values for the deepest node (tau_1) between 0.5 and 1 (different species).
	T1, P1 = get_interspecific_tau(Theta*Theta_A, Theta*Theta_C)

	## Sample Pi values for the second deepest node (tau_2) higher than 0.5 (different species), but lower than tau_1.
	T2, P2 = get_interspecific_tau(Theta*Theta_C, Theta*Theta_E, p_max=P1, T_anc=T1)

	## Sample Pi values for the third deepest node (tau_3) between 1/3 (minimum value -> 0.333) and 0.4 (same species).
	T3, P3 = get_intraspecific_tau(Theta*Theta_E, Theta*Theta_F)

	## Sample the more recent splitting times with a smaller value than the more ancient ones and bigger than 0 (present)
	T4 = np.random.uniform(0, T3)
	T5 = np.random.uniform(0, T4)

	## ms command
	com=subprocess.Popen("./ms %d 1000 -s 1 -t %f -I 6 %d %d %d %d %d %d -n 1 %f -n 2 %f -n 3 %f -n 4 %f -n 5 %f -n 6 %f -ma %s -ej %f 2 1 -ej %f 4 3 -ej %f 6 5 -ej %f 5 3 -ej %f 3 1 -T" % (N_allpops, Theta, N_popA, N_popB, N_popC, N_popD, N_popE, N_popF, Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F, ' '.join(map(str, M.flatten())), T5, T4, T3, T2, T1), shell=True, stdout=subprocess.PIPE).stdout

	output = com.read().splitlines()
	Model_3sp_05.append(np.array(ms2nparray(output)).swapaxes(0,1).reshape(N_allpops,-1).T)
	
	## save parameter values and trees
	par_3sp_05.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (
        Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F,
        T1, T2, T3, T4, T5))
	#Randomly save a number of tree equivalent to the number of traits
	trees_3sp_05.append(random.sample(get_newick(output),100))
	print("Completed %d %% of Model 3sp_05 simulations" % (float(i)/Priorsize*100))

#Save the simulated SNP data
Model_3sp_05=np.array(Model_3sp_05)
np.savez_compressed('trainingSims/Model_3sp_05.npz', Model_3sp_05=Model_3sp_05)
del(Model_3sp_05)

#Save trees from all scenarios
trees=np.concatenate((trees_2sp_0125,trees_3sp_0125,trees_2sp_025,trees_3sp_025,trees_2sp_05,trees_3sp_05),axis=0)
np.savez_compressed('trees.npz', trees=trees)
