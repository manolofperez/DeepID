# ***Euphorbia* scripts and datasets**
This folder contains the empirical datasets of the plant *Euphorbia balsamifera* species complex  
used to test our deep learning approach. 

The folder contains the following files:

simulate_ms_Euphorbia.py - python script to simulate coalescent trees and segregating sites (saved as NumPy arrays)
for the *Euphorbia* dataset.

Train_Test_Predict_Euphorbia.ipynb -  python notebook containing code and outputs for CNN training, cross-validation 
and prediction of the most likely model using empirical data.

Pred_Emp_Comb_BM_Predictions.txt - Predictions for the model trained with the Combined SNPs + BM traits

Pred_Emp_Comb_OU_Predictions.txt - Predictions for the model trained with the Combined SNPs + OU traits

Pred_Emp_SNP_Predictions.txt - Predictions for the model trained with SNPs

Pred_Emp_traits_BM_Predictions.txt - Predictions for the model trained with the BM traits

Pred_Emp_traits_OU_Predictions.txt - Predictions for the model trained with the OU traits

input_SNPs.txt - Segregating sites from the empirical dataset.

input_traits_ade.txt - Traits for the *Euphorbia balsamifera* subsp. *adenensis*

input_traits_bal.txt - Traits for the *Euphorbia balsamifera* subsp. *balsamifera*

input_traits_sep.txt - Traits for the *Euphorbia balsamifera* subsp. *sepium*

Extract_SNPs.R - R script to extract the single SNP from each locus having the fewest missing values.
