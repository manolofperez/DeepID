#!/usr/bin/python3

## in order to use this code you have to have ms installed on your computer
## ms can be freely downloaded from:
## http://home.uchicago.edu/rhudson1/source/mksamples.html

#Import required libraries
import random
import os
import math
import shlex, subprocess
import numpy as np

##define a function to read ms' simulations and transform them into a NumPy array.    
def ms2nparray(xfile):
	g = list(xfile)
	k = [idx for idx,i in enumerate(g) if len(i) > 0 and i.startswith(b'//')]
	f = []
	for i in k:
		L = g[i+5:i+N_allpops+5]
		q = []
		for i in L:
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

## sample size of Euphorbia balsamifera subsp. adenensis.
N_adenensis = 19
## sample size of Euphorbia balsamifera subsp. balsamifera.
N_balsamifera = 80
## sample size of Euphorbia balsamifera subsp. sepium.
N_sepium = 10

## sample size for all pops combined.
N_allpops = N_adenensis + N_balsamifera + N_sepium

## create files to store parameters, trees and the models
os.mkdir("trainingSims")
par_1sp = open("par_1sp.txt","w")
par_2spMorph = open("par_2spMorph.txt","w")
par_2spPhylo = open("par_2spPhylo.txt","w")
par_3sp = open("par_3sp.txt","w")
par_3sp_M = open("par_3sp_M.txt","w")

## create lists to store trees from each scenario
trees_1sp = []
trees_2spMorph = []
trees_2spPhylo = []
trees_3sp = []
trees_3sp_M = []

## create lists to store simulations from each scenario
Model_1sp = []
Model_2spMorph = []
Model_2spPhylo = []
Model_3sp = []
Model_3sp_M = []

####Simulate the species delimitation scenarios####

### Scenario 1: All populations belong to one species
for i in range(Priorsize):
	### Define parameters
    ## Theta (4Neu) values between 2 and 5 according to Rincon-Barrado et al (2024), divided by the size of the alignments (1700 bp)
	Theta = random.uniform(2,5)/1700
	#Now sample relative Thetas (pop sizes) for every deme.
	Theta_A, Theta_B, Theta_S = np.random.uniform(0.5, 2, size=3)

	#And sample migration rates between Nm = 1 and 5 for all of the demes.
	pops = ['A','B','S']
	n = len(pops)

	# draw every possible rate (including i→i) between Nm = 1 and 5 (ms requires migration rates as 4Nm, so bounds are 4 and 20)
	M = np.random.uniform(4, 20, size=(n, n))
	# zero out self-migration
	np.fill_diagonal(M, 0)	

	## Sample Pi values for the deepest node (Pi1_A and Pi1_S) between 1/3 (minimum value -> 0.333) and 0.4 (same species).
	T1, P1 = get_intraspecific_tau(Theta*Theta_A, Theta*Theta_S)

	## Sample Pi values for the second deepest node (Pi2) with a smaller value than deepest one (Pi1).
	T2 = np.random.uniform(0, T1)

	## ms command
	com=subprocess.Popen("./ms %d 428 -s 1 -t %f -I 3 %d %d %d -n 1 %f -n 2 %f -n 3 %f -ma %s -ej %f 2 1 -ej %f 1 3 -T" % (N_allpops, Theta, N_adenensis, N_balsamifera, N_sepium, Theta_A, Theta_B, Theta_S, ' '.join(map(str, M.flatten())), T2, T1), shell=True, stdout=subprocess.PIPE).stdout
	output = com.read().splitlines()
	Model_1sp.append(np.array(ms2nparray(output)).swapaxes(0,1).reshape(N_allpops,-1).T)

	## save parameter values and trees
	par_1sp.write("%f\t%f\t%f\t%f\t%f\t%f\n" % (Theta,Theta_A, Theta_B, Theta_S,T1,T2))
	#Randomly save a number of tree equivalent to the number of traits
	trees_1sp.append(random.sample(get_newick(output),4))
	print("Completed %d %% of Model 1 simulations" % (float(i)/Priorsize*100))

#Save the simulated SNP data
Model_1sp=np.array(Model_1sp)
np.savez_compressed('trainingSims/Model_1sp.npz', Model_1sp=Model_1sp)
del(Model_1sp)

### Two species, morphology
for i in range(Priorsize):

	### Define parameters
    ## Theta (4Neu) values between 2 and 5 according to Rincon-Barrado et al (2024), divided by the size of the alignments (1700 bp)
	Theta = random.uniform(2,5)/1700
	#Now sample relative Thetas (pop sizes) for every deme.
	Theta_A, Theta_B, Theta_S = np.random.uniform(0.5, 2, size=3)

	#And sample migration rates between Nm = 1 and 5 for all of the demes.
	pops = ['A','B','S']
	n = len(pops)

	# draw every possible rate (including i→i) between Nm = 1 and 5 (ms requires migration rates as 4Nm, so bounds are 4 and 20)
	M = np.random.uniform(4, 20, size=(n, n))
	# zero out self-migration
	np.fill_diagonal(M, 0)	

	#Now change it for different species (A,BS) with a value of m (Nm) = 0.1; 4Nm = 0.4
	for i in [0]:
		for j in [1,2]:
			M[i, j] = np.random.uniform(0, 0.4)
			M[j, i] = np.random.uniform(0, 0.4) 

	## Sample Pi values for the deepest node (Pi1_A and Pi1_S) between 0.5 and 1 (different species).
	T1, P1 = get_interspecific_tau(Theta*Theta_A, Theta*Theta_S)

	## Sample Pi values for the second node (Pi1_B and Pi1_S) between 1/3 (minimum value -> 0.333) and 0.4 (same species).
	T2, P2 = get_intraspecific_tau(Theta*Theta_B, Theta*Theta_S)
	
	## ms command
	com=subprocess.Popen("./ms %d 428 -s 1 -t %f -I 3 %d %d %d -n 1 %f -n 2 %f -n 3 %f -ma %s -ej %f 2 3 -ej %f 1 3 -T" % (N_allpops, Theta, N_adenensis, N_balsamifera, N_sepium, Theta_A, Theta_B, Theta_S, ' '.join(map(str, M.flatten())), T2, T1), shell=True, stdout=subprocess.PIPE).stdout
	
	output = com.read().splitlines()
	Model_2spMorph.append(np.array(ms2nparray(output)).swapaxes(0,1).reshape(N_allpops,-1).T)

	## save parameter values and trees
	par_2spMorph.write("%f\t%f\t%f\t%f\t%f\t%f\n" % (Theta,Theta_A, Theta_B, Theta_S,T1,T2))
	#Randomly save a number of tree equivalent to the number of traits
	trees_2spMorph.append(random.sample(get_newick(output),4))
	print("Completed %d %% of Model 2 simulations" % (float(i)/Priorsize*100))

#Save the simulated SNP data
Model_2spMorph=np.array(Model_2spMorph)
np.savez_compressed('trainingSims/Model_2spMorph.npz', Model_2spMorph=Model_2spMorph)
del(Model_2spMorph)

### Two species, Phylogenomics
for i in range(Priorsize):

	### Define parameters
    ## Theta (4Neu) values between 2 and 5 according to Rincon-Barrado et al (2024), divided by the size of the alignments (1700 bp)
	Theta = random.uniform(2,5)/1700
	#Now sample relative Thetas (pop sizes) for every deme.
	Theta_A, Theta_B, Theta_S = np.random.uniform(0.5, 2, size=3)

	#And sample migration rates between Nm = 1 and 5 for all of the demes.
	pops = ['A','B','S']
	n = len(pops)

	# draw every possible rate (including i→i) between Nm = 1 and 5 (ms requires migration rates as 4Nm, so bounds are 4 and 20)
	M = np.random.uniform(4, 20, size=(n, n))
	# zero out self-migration
	np.fill_diagonal(M, 0)	

	#Now change it for different species (AB,S) with a value of m (Nm) = 0.1; 4Nm = 0.4
	for i in [0,1]:
		for j in [2]:
			M[i, j] = np.random.uniform(0, 0.4)
			M[j, i] = np.random.uniform(0, 0.4) 

	## Sample Pi values for the deepest node (Pi1_A and Pi1_S) between 0.5 and 1 (different species).
	T1, P1 = get_interspecific_tau(Theta*Theta_A, Theta*Theta_S)

	## Sample Pi values for the second node (Pi1_A and Pi1_B) between 1/3 (minimum value -> 0.333) and 0.4 (same species).
	T2, P2 = get_intraspecific_tau(Theta*Theta_A, Theta*Theta_B)

	## ms command
	com=subprocess.Popen("./ms %d 428 -s 1 -t %f -I 3 %d %d %d -n 1 %f -n 2 %f -n 3 %f -ma %s -ej %f 2 1 -ej %f 1 3 -T" % (N_allpops, Theta, N_adenensis, N_balsamifera, N_sepium, Theta_A, Theta_B, Theta_S, ' '.join(map(str, M.flatten())), T2, T1), shell=True, stdout=subprocess.PIPE).stdout
	
	## ms command
	output = com.read().splitlines()
	Model_2spPhylo.append(np.array(ms2nparray(output)).swapaxes(0,1).reshape(N_allpops,-1).T)
	
	## save parameter values and trees
	par_2spPhylo.write("%f\t%f\t%f\t%f\t%f\t%f\n" % (Theta,Theta_A, Theta_B, Theta_S,T1,T2))
	#Randomly save a number of tree equivalent to the number of traits
	trees_2spPhylo.append(random.sample(get_newick(output),4))
	print("Completed %d %% of Model 3 simulations" % (float(i)/Priorsize*100))

#Save the simulated SNP data
Model_2spPhylo=np.array(Model_2spPhylo)
np.savez_compressed('trainingSims/Model_2spPhylo.npz', Model_2spPhylo=Model_2spPhylo)
del(Model_2spPhylo)


### Three species
for i in range(Priorsize):

	### Define parameters
    ## Theta (4Neu) values between 2 and 5 according to Rincon-Barrado et al (2024), divided by the size of the alignments (1700 bp)
	Theta = random.uniform(2,5)/1700
	#Now sample relative Thetas (pop sizes) for every deme.
	Theta_A, Theta_B, Theta_S = np.random.uniform(0.5, 2, size=3)

	#And sample migration rates between Nm = 1 and 5 for all of the demes.
	pops = ['A','B','S']
	n = len(pops)

	# draw every possible rate (including i→i) between Nm = 1 and 5 (ms requires migration rates as 4Nm, so bounds are 4 and 20)
	M = np.random.uniform(4, 20, size=(n, n))
	# zero out self-migration
	np.fill_diagonal(M, 0)	

	species = [
    [0],  # A
    [1],  # B
    [2],  # S
	]
	
	#Now change it for different species (A,B,S) with a value of m (Nm) = 0.1; 4Nm = 0.4
	for sp in species:
		for i in sp:
			for j in range(n):
				if j not in sp:
					M[i, j] = np.random.uniform(0, 0.4)
					M[j, i] = np.random.uniform(0, 0.4)

	## Sample Pi values for the deepest node (Pi1_A and Pi1_S) between 0.5 and 1 (different species).
	T1, P1 = get_interspecific_tau(Theta*Theta_A, Theta*Theta_S)

	## Sample Pi values for the second node (Pi1_A and Pi1_B) higher than 0.5 (different species).
	T2, P2 = get_interspecific_tau(Theta*Theta_A, Theta*Theta_B, p_max=P1, T_anc=T1)

	## ms command
	com=subprocess.Popen("./ms %d 428 -s 1 -t %f -I 3 %d %d %d -n 1 %f -n 2 %f -n 3 %f -ma %s -ej %f 2 1 -ej %f 1 3 -T" % (N_allpops, Theta, N_adenensis, N_balsamifera, N_sepium, Theta_A, Theta_B, Theta_S, ' '.join(map(str, M.flatten())), T2, T1), shell=True, stdout=subprocess.PIPE).stdout

	output = com.read().splitlines()
	Model_3sp.append(np.array(ms2nparray(output)).swapaxes(0,1).reshape(N_allpops,-1).T)
	
	## save parameter values and trees
	par_3sp.write("%f\t%f\t%f\t%f\t%f\t%f\n" % (Theta,Theta_A, Theta_B, Theta_S,T1,T2))
	#Randomly save a number of tree equivalent to the number of traits
	trees_3sp.append(random.sample(get_newick(output),4))
	print("Completed %d %% of Model 4 simulations" % (float(i)/Priorsize*100))

#Save the simulated SNP data
Model_3sp=np.array(Model_3sp)
np.savez_compressed('trainingSims/Model_3sp.npz', Model_3sp=Model_3sp)

### Three species with high migration after divergence
for i in range(Priorsize):

	### Define parameters
    ## Theta (4Neu) values between 2 and 5 according to Rincon-Barrado et al (2024), divided by the size of the alignments (1700 bp)
	Theta = random.uniform(2,5)/1700
	#Now sample relative Thetas (pop sizes) for every deme.
	Theta_A, Theta_B, Theta_S = np.random.uniform(0.5, 2, size=3)

	#And sample migration rates between Nm = 1 and 5 for all of the demes.
	pops = ['A','B','S']
	n = len(pops)

	# draw every possible rate (including i→i) between Nm = 1 and 5 (ms requires migration rates as 4Nm, so bounds are 4 and 20)
	M = np.random.uniform(4, 20, size=(n, n))
	# zero out self-migration
	np.fill_diagonal(M, 0)	

	species = [
    [0],  # A
    [1],  # B
    [2],  # S
	]
	
	#Now change it for different species (A,B,S) with a value of m (Nm) = 0.1; 4Nm = 0.4
	for sp in species:
		for i in sp:
			for j in range(n):
				if j not in sp:
					M[i, j] = np.random.uniform(0, 0.4)
					M[j, i] = np.random.uniform(0, 0.4)


	# Now create a migration matrix with higher rates between species after the divergence (reduced after a period of time - up to 1/2 of the divergence time between species.)
	# draw every possible rate (including i→i) between Nm = 0.1 to 0.5 (ms requires migration rates as 4Nm, so bounds are 0.4 and 2)
	
	M_div_2 = np.array(M)

	#Now change it for (A,S) with a value of m (Nm) from 0.1 to 0.5; 4Nm = [0.4,2]
	for i in [0]:
		for j in [1]:
			M_div_2[i, j] = np.random.uniform(0.4,2)
			M_div_2[j, i] = np.random.uniform(0.4,2)

	# Now create a migration matrix with higher rates betwee species after the divergence (reduced after a period of time - up to 1/2 of the divergence time between species.)
	
	# draw every possible rate (including i→i) between Nm = 1 and 5 (ms requires migration rates as 4Nm, so bounds are 4 and 20)
	M_div_1 = np.array(M_div_2)	

	#Now change it for (A,B) with a value of m (Nm) from 0.1 to 0.5; 4Nm = [0.4,2]
	for i in [0]:
		for j in [2]:
			M_div_1[i, j] = np.random.uniform(0.4,2)
			M_div_1[j, i] = np.random.uniform(0.4,2)
			
	## Sample Pi values for the deepest node (Pi1_A and Pi1_S) between 0.5 and 1 (different species).
	T1, P1 = get_interspecific_tau(Theta*Theta_A, Theta*Theta_S)

	## Sample Pi values for the second node (Pi1_A and Pi1_B) higher than 0.5 (different species).
	T2, P2 = get_interspecific_tau(Theta*Theta_A, Theta*Theta_B, p_max=P1, T_anc=T1)

	T2_M = np.random.uniform((T2/2),T2)
	T1_M = np.random.uniform((T1/2),T1)
	T1_M = (T1_M if T1_M > T2 else T2)

	## ms command
	com=subprocess.Popen("./ms %d 428 -s 1 -t %f -I 3 %d %d %d -n 1 %f -n 2 %f -n 3 %f -ma %s -ema %f 3 %s -ej %f 2 1 -ema %f 3 %s -ej %f 1 3 -T" % (N_allpops, Theta, N_adenensis, N_balsamifera, N_sepium, Theta_A, Theta_B, Theta_S, ' '.join(map(str, M.flatten())), T2_M, ' '.join(map(str, M_div_2.flatten())), T2, T1_M, ' '.join(map(str, M_div_1.flatten())), T1), shell=True, stdout=subprocess.PIPE).stdout

	output = com.read().splitlines()
	Model_3sp_M.append(np.array(ms2nparray(output)).swapaxes(0,1).reshape(N_allpops,-1).T)
	
	## save parameter values and trees
	par_3sp_M.write("%f\t%f\t%f\t%f\t%f\t%f\n" % (Theta,Theta_A, Theta_B, Theta_S,T1,T2))
	#Randomly save a number of tree equivalent to the number of traits
	trees_3sp_M.append(random.sample(get_newick(output),4))
	print("Completed %d %% of Model 5 simulations" % (float(i)/Priorsize*100))

#Save the simulated SNP data
Model_3sp_M=np.array(Model_3sp_M)
np.savez_compressed('trainingSims/Model_3sp_M.npz', Model_3sp_M=Model_3sp_M)

#Save trees from all scenarios
trees=np.concatenate((trees_1sp,trees_2spMorph,trees_2spPhylo,trees_3sp,trees_3sp_M),axis=0)
np.savez_compressed('trees.npz', trees=trees)
