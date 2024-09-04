import csv
import random
import itertools
import numpy as np
import psutil

from numpy.random import multivariate_normal as multinorm
from random import seed
from fastkde.fastKDE import pdf_at_points
from multiprocessing import Pool

parent = psutil.Process()
parent.nice(psutil.IDLE_PRIORITY_CLASS)


# Define DGP function

def dgp(n, gamma, beta2, beta3, rho):
	
	# Assume X ~ multivariate normal distribution
	
    x_mean = (0, 0)
    x_cov = [ [1,0], [0,1] ]
    x = multinorm(x_mean, x_cov, n ) 

    x3 = x[ : , 1]
    x2 = x[ : , 0]

	# If X ~ chi^2 distribution
    #x3 = chi2.rvs(loc=0, scale=0.1, df=1, size = n)
    #x2 = chi2.rvs(loc=0, scale=0.1, df=1, size = n)

    eps_mean = [0, 0, 0]
    eps_cov = [ [1, rho, rho], [rho, 1, rho], [rho, rho, 1] ]
    eps = multinorm(eps_mean, eps_cov, n)

    # binary Y3
    y3 = np.array( x3 + eps[ : ,2] > 0, dtype=int)

    # Note that gamma * beta2 = 0 => gamma = 0 or beta2 = 0 
	
	# Select the variable type of Y2 and Y1 below

    # binary Y2
    y2 = np.array( x2 + gamma * y3  + eps[ : ,1] > 0, dtype=int )

    # continuous Y2
    #y2 = np.array( x2 + gamma * y3  + eps[ : ,1]  )

    # binary Y1
    #y1 = np.array( beta2 * y2 + beta3 * y3  + eps[ : ,0] > 0, dtype=int )

    # continuous Y1
    y1 = beta2 * y2 + beta3 * y3  + eps[ : ,0]

    
    return(x2, x3, y1, y2, y3)
	

# Define tau_data function

def tau_data(x2, x3, y1, y2, y3):
    
    n = len(y2) 
    
    comb_index = list( itertools.permutations( range(n), 2) )
    comb_index = np.asarray(comb_index)

    i = comb_index[ : , 0]
    j = comb_index[ : , 1]
    
    y2ij = y2[i] - y2[j]
    x3ij = x3[i] - x3[j]
    x2ij = x2[i] - x2[j]
    
    ind_y3ij = np.where( y3[i] == y3[j] )[0]

    n_y3ij = len(ind_y3ij)
        
    y1ij = (y1[i] - y1[j])[ind_y3ij]
    x2ij_mat = (x2[i] - x2[j])[ind_y3ij]
      
    return(y2ij, x3ij, x2ij, y1ij, x2ij_mat, n_y3ij )


seed(69434)

n = 1000
block_size = n

nboots = 100

cpu_iters = 10
nsim = 25
processes = nsim # maximum 40 cores



# new csv file

fields = ['T32', 'T21', 'T32_se', 'T21_se']

with open('July11_2024_binY2_contY1_n1000_B100_g09_b09_rho07.csv', 'w', newline='') as newcsv:
    July11_2024_binY2_contY1_n1000_B100_g09_b09_rho07 = csv.writer(newcsv)
    July11_2024_binY2_contY1_n1000_B100_g09_b09_rho07.writerow(fields)
newcsv.close()



# Define task

def task(sim):

    print('CPU no', sim, 'is running!!')
    
    for i in range(cpu_iters):

        dta = dgp( n = n, gamma = 0.9, beta2 = 0.9, beta3 = 1, rho = 0.7 )
    
        x2 = dta[0] ; x3 = dta[1]
        y1 = dta[2] ; y2 = dta[3] ; y3 = dta[4]

        tau_dta = tau_data(x2, x3, y1, y2, y3)
        y2ij = tau_dta[0]
        x3ij = tau_dta[1]
        x2ij = tau_dta[2]
        y1ij = tau_dta[3]
        x2ij_mat = tau_dta[4]
        n_y3ij = tau_dta[5]

        w_hat_ij = pdf_at_points(x2ij)

        T32 = np.dot( y2ij * np.sign(x3ij) , w_hat_ij) / np.sum(w_hat_ij)
        T21 = np.dot( y1ij , np.sign(x2ij_mat) ) / n_y3ij

        # Initialize bootstrap method
    
        T32_bt = []
        T21_bt = []
    
        for B in range(nboots):
                
            # m out of n bootstrap
            bt_ind = random.choices( range(n), k = block_size )
        
		
            x2_bt = x2[bt_ind] ; x3_bt = x3[bt_ind]
            y1_bt = y1[bt_ind] ; y2_bt = y2[bt_ind] ; y3_bt = y3[bt_ind]
        
            tau_dta_bt = tau_data(x2_bt, x3_bt, y1_bt, y2_bt, y3_bt)
            y2ij_bt = tau_dta_bt[0]
            x3ij_bt = tau_dta_bt[1]
            x2ij_bt = tau_dta_bt[2]
            y1ij_bt = tau_dta_bt[3]
            x2ij_mat_bt = tau_dta_bt[4]
            n_y3ij_bt = tau_dta_bt[5]

            w_hat_ij_bt = pdf_at_points(x2ij_bt)

            t32_bt = np.dot( y2ij_bt * np.sign(x3ij_bt), w_hat_ij_bt) / np.sum(w_hat_ij_bt)
            t21_bt = np.dot( y1ij_bt , np.sign(x2ij_mat_bt) ) / n_y3ij_bt
        
            T32_bt.append(t32_bt)
            T21_bt.append(t21_bt)

            print('CPU no', sim, 'bootstrap no', B, 't32bt=', round(t32_bt,3), 't21bt=', round(t21_bt,3) , '\n' )
        
        # Calculate bootstrap standard deviation
        
        T32_se = np.nanstd(T32_bt, ddof=1) 
        T21_se = np.nanstd(T21_bt, ddof=1)
    
        print('CPU no', sim+1, 'iter', i+1, ':', round(T32,3) , round(T32_se,3), round(T32/T32_se,3), round(T21,3) , round(T21_se,3), round(T21/T21_se,3), '\n')

        output = [T32, T21, T32_se, T21_se]


        # Write the result from this simulation

        with open('July11_2024_binY2_contY1_n1000_B100_g09_b09_rho07.csv', 'a', newline='') as simdta:
            July11_2024_binY2_contY1_n1000_B100_g09_b09_rho07 = csv.writer(simdta)
            July11_2024_binY2_contY1_n1000_B100_g09_b09_rho07.writerow(output)
        simdta.close()
    

if __name__ == '__main__':
    pool = Pool(processes = processes)
    #pool = Pool(processes = 40)                         # Create a multiprocessing Pool
    pool.map(task, range(nsim))                       # process data_inputs iterable with pool
