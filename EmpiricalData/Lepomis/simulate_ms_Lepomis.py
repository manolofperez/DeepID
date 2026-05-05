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
## sample size of Laqu.
N_Laqu = 78*2
## sample size of Lmeg.
N_Lmeg = 47*2
## sample size of Loua.
N_Loua = 10*2
## sample size of Lozk.
N_Lozk = 24*2
## sample size of Lpel.
N_Lpel = 29*2
## sample size of Lsol.
N_Lsol = 41*2

## sample size for all pops combined.
N_allpops = N_Laqu + N_Lmeg + N_Loua + N_Lozk + N_Lpel + N_Lsol

## create files to store parameters, trees and the models
os.mkdir("trainingSims")
par_2sp = open("par_2sp.txt","w")
par_6sp = open("par_6sp.txt","w")
par_6spMig = open("par_6spMig.txt","w")
par_6spMig_all = open("par_6spMig_all.txt","w")
## create lists to store trees from each scenario
trees_2sp = []
trees_6sp = []
trees_6spMig = []
trees_6spMig_all = []

## create lists to store simulations from each scenario
Model_2sp = []
Model_6sp = []
Model_6spMig = []
Model_6spMig_all = []


####Simulate the species delimitation scenarios####

## Scenario 1: Two species, Lpel and all other (note this hypothesis violates the topology recovered in Kim et al. 2022)
for i in range(Priorsize):

	### Define parameters
	## Theta (4Neu) values between 0.001 ans 0.02
	Theta = random.uniform(0.001,0.02)

	#Now sample relative Thetas (pop sizes) for every deme. I will use A-F codes to simplify. THe order is: Laqu, Lmeg, Loua, Lozk, Lpel, Lsol.
	Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F= np.random.uniform(0.5, 2, size=6)

	#And sample migration rates between Nm = 1 and 5 for all of the demes.
	pops = ['A','B','C','D','E','F']
	n = len(pops)

	# draw every possible rate (including i→i) between Nm = 1 and 5 (ms requires migration rates as 4Nm, so bounds are 4 and 20)
	M = np.random.uniform(4, 20, size=(n, n))
	# zero out self-migration
	np.fill_diagonal(M, 0)

	#Now change it for different species (ABCDF,E) with a value of m (Nm) = 0.1; 4Nm = 0.4
	for i in [0,1,2,3,5]:
		for j in [4]:
			M[i, j] = np.random.uniform(0, 0.4)
			M[j, i] = np.random.uniform(0, 0.4)

	## Sample P values for the deepest node (Pi1_A and Pi1_E) between 0.5 and 1 (different species).
	T1, P1 = get_interspecific_tau(Theta*Theta_A, Theta*Theta_E)
	## Transform divergence times to 4Ne generations units (required by ms)

	## Sample P values for the second deepest node (tau2) between 0.333 and 0.44 (same species).
	T2, P2 = get_intraspecific_tau(Theta*Theta_A, Theta*Theta_F)

	## Sample the more recent splitting times with a smaller value than the more ancient ones and bigger than 0 (present)
	T3 = np.random.uniform(0, T2)
	T4 = np.random.uniform(0, T3)
	T5 = np.random.uniform(0, T4)

	## ms command
	com=subprocess.Popen("./ms %d 1000 -s 1 -t %f -I 6 %d %d %d %d %d %d -n 1 %f -n 2 %f -n 3 %f -n 4 %f -n 5 %f -n 6 %f -ma %s -ej %f 4 2 -ej %f 2 3 -ej %f 3 6 -ej %f 6 1 -ej %f 5 1 -T" % (N_allpops, Theta, N_Laqu, N_Lmeg, N_Loua, N_Lozk, N_Lpel, N_Lsol, Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F, ' '.join(map(str, M.flatten())), T5, T4, T3, T2, T1), shell=True, stdout=subprocess.PIPE).stdout
	# read ms output
	output = com.read().splitlines()
	# save the SNPs output as a NumPy array
	Model_2sp.append(np.array(ms2nparray(output)).swapaxes(0,1).reshape(N_allpops,-1).T)

	## save parameter values
	par_2sp.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (
	Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F,
	T1, T2, T3, T4, T5))
	#Randomly save a number of tree equivalent to the number of traits
	trees_2sp.append(random.sample(get_newick(output),28))
	print("Completed %d %% of Model 1 simulations" % (float(i)/Priorsize*100))

#Compress and save the simulated SNP data
Model_2sp=np.array(Model_2sp)
np.savez_compressed('trainingSims/Model_2sp.npz', Model_2sp=Model_2sp)
del(Model_2sp)
np.savez_compressed('trees_2sp.npz', trees_2sp=trees_2sp)

### Scenario 2: Six species
for i in range(Priorsize):

	### Define parameters
	## Theta (4Neu) values between 0.001 ans 0.02
	Theta = random.uniform(0.001,0.02)

	#Now sample relative Thetas (pop sizes) for every deme. I will use A-F codes to simplify. THe order is: Laqu, Lmeg, Loua, Lozk, Lpel, Lsol.
	Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F= np.random.uniform(0.5, 2, size=6)

	#And sample migration rates between Nm = 1 and 5 for all of the demes.
	pops = ['A','B','C','D','E','F']
	n = len(pops)

	# draw every possible rate (including i→i) between Nm = 1 and 5 (ms requires migration rates as 4Nm, so bounds are 4 and 20)
	M = np.random.uniform(4, 20, size=(n, n))
	# zero out self-migration
	np.fill_diagonal(M, 0)

	species = [
    [0],
    [1],
    [2],
    [3],
    [4],
    [5],
	]
	
	#Now change it for different species with a value of m (Nm) = 0.1; 4Nm = 0.4
	for sp in species:
		for i in sp:
			for j in range(n):
				if j not in sp:
					M[i, j] = np.random.uniform(0, 0.4)
					M[j, i] = np.random.uniform(0, 0.4)

	## Sample P values for the deepest node (Pi1_A and Pi1_F) between 0.5 and 1 (different species).
	T1, P1 = get_interspecific_tau(Theta*Theta_A, Theta*Theta_F)
	## Transform divergence times to 4Ne generations units (required by ms)

	## Sample P values for the second deepest node (tau2) between 0.5 and the P fro the previous node (different species).
	T2, P2 = get_interspecific_tau(Theta*Theta_C, Theta*Theta_F, p_max=P1, T_anc=T1)

	## Sample P values for the third deepest node (tau3) between 0.5 and the P fro the previous node (different species).
	T3, P3 = get_interspecific_tau(Theta*Theta_B, Theta*Theta_C, p_max=P2, T_anc=T2)

	## Sample P values for the fourth deepest node (tau4) between 0.5 and the P fro the previous node (different species).
	T4, P4 = get_interspecific_tau(Theta*Theta_B, Theta*Theta_D, p_max=P3, T_anc=T3)

	## Sample P values for the fifth deepest node (tau5) between 0.5 and the P fro the previous node (different species).
	T5, P5 = get_interspecific_tau(Theta*Theta_D, Theta*Theta_E, p_max=P4, T_anc=T4)

	## ms command
	com=subprocess.Popen("./ms %d 1000 -s 1 -t %f -I 6 %d %d %d %d %d %d -n 1 %f -n 2 %f -n 3 %f -n 4 %f -n 5 %f -n 6 %f -ma %s -ej %f 5 4 -ej %f 4 2 -ej %f 2 3 -ej %f 3 6 -ej %f 6 1 -T" % (N_allpops, Theta, N_Laqu, N_Lmeg, N_Loua, N_Lozk, N_Lpel, N_Lsol, Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F, ' '.join(map(str, M.flatten())), T5, T4, T3, T2, T1), shell=True, stdout=subprocess.PIPE).stdout
	# read ms output
	output = com.read().splitlines()
	# save the SNPs output as a NumPy array
	Model_6sp.append(np.array(ms2nparray(output)).swapaxes(0,1).reshape(N_allpops,-1).T)

	## save parameter values
	par_6sp.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (
        Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F,
        T1, T2, T3, T4, T5))
	#Randomly save a number of tree equivalent to the number of traits
	trees_6sp.append(random.sample(get_newick(output),28))
	print("Completed %d %% of Model 2 simulations" % (float(i)/Priorsize*100))

#Compress and save the simulated SNP data
Model_6sp=np.array(Model_6sp)
np.savez_compressed('trainingSims/Model_6sp.npz', Model_6sp=Model_6sp)
del(Model_6sp)
np.savez_compressed('trees_6sp.npz', trees_6sp=trees_6sp)

### Six species + M between putative hybrids
for i in range(Priorsize):

	### Define parameters
	## Theta (4Neu) values between 0.001 ans 0.02
	Theta = random.uniform(0.001,0.02)

	#Now sample relative Thetas (pop sizes) for every deme. I will use A-F codes to simplify. THe order is: Laqu, Lmeg, Loua, Lozk, Lpel, Lsol.
	Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F= np.random.uniform(0.5, 2, size=6)

	#And sample migration rates between Nm = 1 and 5 for all of the demes.
	pops = ['A','B','C','D','E','F']
	n = len(pops)

	# draw every possible rate (including i→i) between Nm = 1 and 5 (ms requires migration rates as 4Nm, so bounds are 4 and 20)
	M = np.random.uniform(4, 20, size=(n, n))
	# zero out self-migration
	np.fill_diagonal(M, 0)

	species = [
    [0],
    [1],
    [2],
    [3],
    [4],
    [5],
	]
	
	#Now change it for different species (A,B,S) with a value of m (Nm) = 0.1; 4Nm = 0.4
	for sp in species:
		for i in sp:
			for j in range(n):
				if j not in sp:
					M[i, j] = np.random.uniform(0, 0.4)
					M[j, i] = np.random.uniform(0, 0.4)

	#Now change only for species pais with putative hybrids Laqu, Lmeg, Loua, Lozk, Lpel, Lsol
	#-m 1 3 %f -m 1 5 %f -m 2 6 %f -m 4 1 %f -m 4 2 %f -m 5 2 %f -m 6 2 %f
	#Loua->Laqu
	M[0, 2] = np.random.uniform(0.4,2)
	#Lpel->Laqu
	M[0, 4] = np.random.uniform(0.4,2)
	#Lsol->Lmeg
	M[1, 5] = np.random.uniform(0.4,2)
	#Laqu->Lozk
	M[3, 0] = np.random.uniform(0.4,2)
	#Lmeg->Lozk
	M[3, 1] = np.random.uniform(0.4,2)
	#Lmeg->Lpel
	M[4, 1] = np.random.uniform(0.4,2)
	#Lmeg->Lsol
	M[5, 1] = np.random.uniform(0.4,2)


	## Sample P values for the deepest node (Pi1_A and Pi1_F) between 0.5 and 1 (different species).
	T1, P1 = get_interspecific_tau(Theta*Theta_A, Theta*Theta_F)
	## Transform divergence times to 4Ne generations units (required by ms)

	## Sample P values for the second deepest node (tau2) between 0.5 and the P fro the previous node (different species).
	T2, P2 = get_interspecific_tau(Theta*Theta_C, Theta*Theta_F, p_max=P1, T_anc=T1)

	## Sample P values for the third deepest node (tau3) between 0.5 and the P fro the previous node (different species).
	T3, P3 = get_interspecific_tau(Theta*Theta_B, Theta*Theta_C, p_max=P2, T_anc=T2)

	## Sample P values for the fourth deepest node (tau4) between 0.5 and the P fro the previous node (different species).
	T4, P4 = get_interspecific_tau(Theta*Theta_B, Theta*Theta_D, p_max=P3, T_anc=T3)

	## Sample P values for the fifth deepest node (tau5) between 0.5 and the P fro the previous node (different species).
	T5, P5 = get_interspecific_tau(Theta*Theta_D, Theta*Theta_E, p_max=P4, T_anc=T4)

	## ms command
	com=subprocess.Popen("./ms %d 1000 -s 1 -t %f -I 6 %d %d %d %d %d %d -n 1 %f -n 2 %f -n 3 %f -n 4 %f -n 5 %f -n 6 %f -ma %s -ej %f 5 4 -ej %f 4 2 -ej %f 2 3 -ej %f 3 6 -ej %f 6 1 -T" % (N_allpops, Theta, N_Laqu, N_Lmeg, N_Loua, N_Lozk, N_Lpel, N_Lsol, Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F, ' '.join(map(str, M.flatten())), T5, T4, T3, T2, T1), shell=True, stdout=subprocess.PIPE).stdout

	# read ms output
	output = com.read().splitlines()
	# save the SNPs output as a NumPy array
	Model_6spMig.append(np.array(ms2nparray(output)).swapaxes(0,1).reshape(N_allpops,-1).T)

	## save parameter values
	par_6spMig.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (
        Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F,
        T1, T2, T3, T4, T5))
	#Randomly save a number of tree equivalent to the number of traits
	trees_6spMig.append(random.sample(get_newick(output),28))
	print("Completed %d %% of Model 3 simulations" % (float(i)/Priorsize*100))

#Compress and save the simulated SNP data
Model_6spMig=np.array(Model_6spMig)
np.savez_compressed('trainingSims/Model_6spMig.npz', Model_6spMig=Model_6spMig)
del(Model_6spMig)
np.savez_compressed('trees_6spMig.npz', trees_6spMig=trees_6spMig)

### Six species + M between all pairs
for i in range(Priorsize):

	### Define parameters
	## Theta (4Neu) values between 0.001 ans 0.02
	Theta = random.uniform(0.001,0.02)

	#Now sample relative Thetas (pop sizes) for every deme. I will use A-F codes to simplify. THe order is: Laqu, Lmeg, Loua, Lozk, Lpel, Lsol.
	Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F= np.random.uniform(0.5, 2, size=6)

	#And sample migration rates between Nm = 1 and 5 for all of the demes.
	pops = ['A','B','C','D','E','F']
	n = len(pops)

	# draw every possible rate (including i→i) between Nm = 1 and 5 (ms requires migration rates as 4Nm, so bounds are 4 and 20)
	M = np.random.uniform(4, 20, size=(n, n))
	# zero out self-migration
	np.fill_diagonal(M, 0)

	species = [
    [0],
    [1],
    [2],
    [3],
    [4],
    [5],
	]
	
	#Now change it for different species (A,B,S) with a value of m (Nm) = [0.1,0.5]; 4Nm = [0.4,2]
	for sp in species:
		for i in sp:
			for j in range(n):
				if j not in sp:
					M[i, j] = np.random.uniform(0.4,2)
					M[j, i] = np.random.uniform(0.4,2)

	## Sample P values for the deepest node (Pi1_A and Pi1_F) between 0.5 and 1 (different species).
	T1, P1 = get_interspecific_tau(Theta*Theta_A, Theta*Theta_F)
	## Transform divergence times to 4Ne generations units (required by ms)

	## Sample P values for the second deepest node (tau2) between 0.5 and the P fro the previous node (different species).
	T2, P2 = get_interspecific_tau(Theta*Theta_C, Theta*Theta_F, p_max=P1, T_anc=T1)

	## Sample P values for the third deepest node (tau3) between 0.5 and the P fro the previous node (different species).
	T3, P3 = get_interspecific_tau(Theta*Theta_B, Theta*Theta_C, p_max=P2, T_anc=T2)

	## Sample P values for the fourth deepest node (tau4) between 0.5 and the P fro the previous node (different species).
	T4, P4 = get_interspecific_tau(Theta*Theta_B, Theta*Theta_D, p_max=P3, T_anc=T3)

	## Sample P values for the fifth deepest node (tau5) between 0.5 and the P fro the previous node (different species).
	T5, P5 = get_interspecific_tau(Theta*Theta_D, Theta*Theta_E, p_max=P4, T_anc=T4)

	## ms command
	com=subprocess.Popen("./ms %d 1000 -s 1 -t %f -I 6 %d %d %d %d %d %d -n 1 %f -n 2 %f -n 3 %f -n 4 %f -n 5 %f -n 6 %f -ma %s -ej %f 5 4 -ej %f 4 2 -ej %f 2 3 -ej %f 3 6 -ej %f 6 1 -T" % (N_allpops, Theta, N_Laqu, N_Lmeg, N_Loua, N_Lozk, N_Lpel, N_Lsol, Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F, ' '.join(map(str, M.flatten())), T5, T4, T3, T2, T1), shell=True, stdout=subprocess.PIPE).stdout

	# read ms output
	output = com.read().splitlines()
	Model_6spMig_all.append(np.array(ms2nparray(output)).swapaxes(0,1).reshape(N_allpops,-1).T)

	## save parameter values
	par_6spMig_all.write("%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n" % (
        Theta_A, Theta_B, Theta_C, Theta_D, Theta_E, Theta_F,
        T1, T2, T3, T4, T5))
	#Randomly save a number of tree equivalent to the number of traits
	trees_6spMig_all.append(random.sample(get_newick(output),28))
	print("Completed %d %% of Model 4 simulations" % (float(i)/Priorsize*100))

#Compress and save the simulated SNP data
Model_6spMig_all=np.array(Model_6spMig_all)
np.savez_compressed('trainingSims/Model_6spMig_all.npz', Model_6spMig_all=Model_6spMig_all)
del(Model_6spMig_all)
np.savez_compressed('trees_6spMig_all.npz', trees_6spMig_all=trees_6spMig_all)

# Compress and save trees from all scenarios
#trees=np.concatenate((trees_2sp,trees_6sp,trees_6spMig,trees_6spMig_all),axis=0)
#np.savez_compressed('trees.npz', trees=trees)
