# Aldair E. Gongora 
# KABLAB 
# Boston University
# Paper: Designing lattices for impact protection using transfer learning
# Date: August 09, 2021
# Revisition History:
# October 18, 2022 - Added comments to code 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import pickle 
from sklearn.utils import shuffle
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from scipy import stats

plt.close("all")
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['font.family'] = "Arial"
mpl.rcParams['font.weight'] = "bold"
c_belizeblue = [46/255, 226/255, 237/255]
c_livingcoral = [250/255, 114/255, 104/255]

#%% Functions

# sequential learning function
def sequential_learning(X_train_pca,Y_train_pca,group_idx,x_train_pca_mean,y_train_pca_mean,rand_seed):
    # preallocation for learning loop
    coeflist = []
    interceptlist = [] 
    y_test_vec = np.array([])
    rmse_loocv_all_list = np.array([])
    rmse_train_list = np.array([])
    gkf = GroupKFold(n_splits=6)
    
    # get shuffles
    x_train_pca, y_train_pca, Xgroup_idx = shuffle(X_train_pca, Y_train_pca,group_idx.reshape(-1), random_state=rand_seed)
    
    for j in np.arange(11):
        # initialize arrays: coef list, intercept list, prediction list 
        coef_list = np.array([])
        intercept_list = np.array([])
        y_pred_list = np.array([])
        y_test_vec = np.array([])
        rmse_train_i_list  = np.array([])
        
        if j == 0:
            # assign first PC term 
            term_vec = [0]  
        
        # print pc term (with 1 and the first entry and not 0)
        term_vec_print = [x + 1 for x in term_vec ]
        print(term_vec_print)   
        
        i = 0
        for train_index, test_index in gkf.split(x_train_pca, y_train_pca, groups=group_idx.reshape(-1)):
            
            # structure training set
            x_train_i_pca = x_train_pca[train_index,:]
            x_train_i = x_train_i_pca[:,term_vec]
            y_train_i = y_train_pca[train_index]
            # structure testing set
            x_test_i_pca = x_train_pca[test_index,:]
            x_test_pca = x_test_i_pca[:,term_vec]
            y_test_i = y_train_pca[test_index]
            
            # train model  
            model = LinearRegression()
            model.fit(x_train_i, y_train_i)
            
            # compute rmse for training set
            y_pred_train_i = model.predict(x_train_i)
            
            #rmse
            rmse_train_i = np.sqrt(np.mean((y_pred_train_i-y_train_i)**2))
            
            # append rmse to list 
            rmse_train_i_list = np.hstack([rmse_train_i_list,rmse_train_i])
        
            # prediction: for mean  
            y_pred_i = model.predict(x_test_pca)
            
            # save predictions: mean 
            y_pred_list = np.hstack([y_pred_list,y_pred_i]) 
            y_test_vec = np.hstack([y_test_vec,y_test_i])
            
            # get coefficients (betas)
            coef_i = model.coef_
            
            # save coefficients (betas)
            if i == 0: 
                coef_list = np.hstack([coef_list,coef_i])
            else:
                coef_list = np.vstack([coef_list,coef_i])
            
            # get intercept 
            intercept_i = model.intercept_
            
            # save intercept 
            intercept_list = np.hstack([intercept_list,intercept_i])
            
            # update i 
            i = i + 1
        
        # save coefficient as a list 
        coeflist.append(coef_list)
        
        # save intercept as a list 
        interceptlist.append(intercept_list)
        
        # compute rmse loocv based on entire dataset (all) (not mean)
        rmse_loocv_all = np.sqrt(np.mean((y_test_vec.reshape(-1)-y_pred_list.reshape(-1))**2))

        # save rmse loocv based on entire dataset (all) (not mean) 
        rmse_loocv_all_list = np.hstack([rmse_loocv_all_list,rmse_loocv_all]) 
        
        # save rmse for training set 
        rmse_train_list_i = np.mean(rmse_train_i_list)
        rmse_train_list = np.hstack([rmse_train_list,rmse_train_list_i])
                
        # compute R2
        r_list = np.array([])
        p_list = np.array([])
        
        # retrain model and get prediction 
        # train model  
        
        model_delta = LinearRegression()
        
        model_delta.fit(x_train_pca[:,term_vec], y_train_pca)
        
        # prediction: for mean  
        y_pred_delta = model_delta.predict(x_train_pca_mean[:,term_vec])
        
        # compute delta 
        delta =  y_train_pca_mean-y_pred_delta
        
        # save metrics
        for k in np.arange(11):
            x1 = x_train_pca_mean[:,k]
            y1 = delta
            r,p = stats.pearsonr(x1, y1)
            r_list = np.hstack([r_list,r])
            p_list = np.hstack([p_list,p])
            
        # find next PC
        pc_argsort = np.flip(np.argsort(r_list**2))
        
        for pcid in np.arange(len(pc_argsort)):
            pc_next = pc_argsort[pcid]
            pc_next_boolid = (np.array(term_vec)==pc_next)
            if sum(pc_next_boolid)>=1 :
                #print('PC is inlist')
                print('')
            else: 
                # next pc 
                #term_vec.append(pc_next+4)
                term_vec.append(pc_next)
                print(pc_next)
                break

    return term_vec,rmse_loocv_all_list,coeflist,interceptlist,rmse_train_list

#%% User input files: Filenames and preliminaries

# name of pickle file with relevant data
filename_octetdata = "Data_OctetModel.pkl"

# load data
open_filename_octetdata = open(filename_octetdata, "rb")
loaded_list = pickle.load(open_filename_octetdata)
open_filename_octetdata.close()

# load and assign relevant variable names 
x = loaded_list[0]             # Force data
x_12 = loaded_list[1]          # Descriptors of lattice. Column headers: [Unit Cell Identifier,voxel_bend,voxel_stretch,voxel_vert, x_bend,x_stretch,x_vert,x_joint,WADL ID Number]
x_12pca = loaded_list[2]       # Descriptors of lattice + Principal components
x_12pca_acc = loaded_list[3]   # Lattice with acceleration information
x_12_acc_mean = loaded_list[4] # Lattice with mean acceleration information
x_12pca_acc_ITID = loaded_list[5] # Lattice with mean acceleration information. Column headers: [Impact Testing Number,Unit Cell Identifier,voxel_bend,voxel_stretch,voxel_vert,x_12pca]

# number of features for final prediction. This number is based on X. 
num_mdl_feat = 6

# number of simulations 
num_sims = 100

# number of principal components considered
num_pcs = 11

#%% Training set based on number of PCs

# x: relevant PCs and select subset of PCs
x_train_allpca = x_12pca_acc[:,4:-1].astype('float')
x_train_pca_numpcs = x_train_allpca[:,:num_pcs]
x_train_pca = x_train_pca_numpcs

# y: acceleration
y_train_pca = x_12pca_acc[:,-1].astype('float')
y_train_pca_all_exp = x_12pca_acc[:,-1].astype('float')

# group ids to sort through impact experiments (acceleration measurements)
# with similar design variables (3 impact experiments for each design). 
group_idx = np.zeros([len(x_12pca_acc),1])
for i in np.arange(len(x_12_acc_mean)):
    # find index
    idx =(x_12pca_acc[:,0] == x_12_acc_mean[i,0])&(x_12pca_acc[:,1] == x_12_acc_mean[i,1])&(x_12pca_acc[:,2] == x_12_acc_mean[i,2])&(x_12pca_acc[:,3] == x_12_acc_mean[i,3])
    # assign to group 
    group_idx[idx,:] = i

#%% LOOCV training and prediction (iterative)

# desing values (for 3 impact experiments for each design. 18 unique designs)
x_train_pca_mean = x_12_acc_mean[:,4:-1]
x_train_pca_mean = x_train_pca_mean[:,:num_pcs]

# mean acceleration training data: acc
y_train_pca_mean = x_12_acc_mean[:,-1]

# preallocation and random seed for loop 
term_vec_hist = np.zeros([num_sims,num_pcs])
rmse_hist = np.zeros([num_sims,num_pcs])
rand_seed_list = rand_seed_list = np.arange(100,10000+100,100)
np.random.seed(50)
rand_seed_list = np.random.randint(1000000, size=num_sims)
coeflist_hist = []
interceptlist_hist = [] 
rmse_train_list_hist = np.zeros([num_sims,11])

# loop through group shuffles
for randid in np.arange(num_sims):
    
    # select random seed id 
    rand_seed_i = rand_seed_list[randid]
    print(rand_seed_i)
    
    # one shuffle
    term_vecZ,rmse_loocv_all_listZ,coeflist,interceptlist,rmse_train_list = sequential_learning(x_train_pca,y_train_pca,group_idx,x_train_pca_mean,y_train_pca_mean,rand_seed_i)
    
    # save results 
    term_vec_hist[randid,:] = term_vecZ
    rmse_hist[randid,:] = rmse_loocv_all_listZ
    
    # save coefficient as a list 
    coeflist_hist.append(coeflist)
    
    # save intercept as a list 
    interceptlist_hist.append(interceptlist)
    
    # save rmse train list 
    rmse_train_list_hist[randid,:] = rmse_train_list 

#%% Prediction for octets (interpolation)

# mean over terms 
term_vec_hist_mean = np.mean(term_vec_hist,axis=0)

# select relevant pc terms 
num_pcs_id_alt = term_vec_hist_mean[:num_mdl_feat]
num_pcs_id_alt = num_pcs_id_alt.astype('int')
x_octet_input = x_train_pca_mean[:,num_pcs_id_alt]
x_octet_input_all = x_train_pca[:,num_pcs_id_alt]

# prediction: linear regression based on number of terms  

# collect intercept and coef 
coef_mdl_hist = np.zeros([num_sims,num_mdl_feat])
intercept_mdl_hist = np.zeros([num_sims,1])

for coef_id in np.arange(num_sims):
    # get coefficient for that simulation 
    coef_mdl = coeflist_hist[coef_id][num_mdl_feat-1]
    # get intercept for that simulation 
    intercept_mdl = interceptlist_hist[coef_id][num_mdl_feat-1]
    # save coefficient
    coef_mdl_hist[coef_id,:] = np.mean(coef_mdl,axis=0)
    # save intercept 
    intercept_mdl_hist[coef_id] = np.mean(intercept_mdl,axis=0)

# mean over simulations 
coef_octa = np.mean(coef_mdl_hist,axis=0)
intercept_octa = np.mean(intercept_mdl_hist)

# y output: prediction for mean
y_octet_pred = np.matmul(x_octet_input,coef_octa) + intercept_octa

#%% Figure: parity plot

plt.figure()
plt.plot(y_train_pca_mean,y_octet_pred,markersize=7*3,color='None',markerfacecolor=[128/255,128/255,128/255],markeredgecolor='k',marker='o',alpha=0.7,zorder=10)
plt.plot(np.arange(300,1000),np.arange(300,1000),color='k',linewidth=2,zorder=5)

plt.xticks(np.arange(300.,700,100),fontsize=14,weight='bold')
plt.yticks(np.arange(300.,700,100),fontsize=14,weight='bold')
plt.xlim(300,600)
plt.ylim(300,600)
plt.xlabel("$a_{m} (g)$",fontsize=18,weight='bold')
plt.ylabel("$\hat{a}_{m} (g)$",fontsize=18,weight='bold')
plt.tick_params(direction="in",width=1.,size=2.5)
plt.rcParams['savefig.facecolor']='white'
plt.tight_layout(pad=0.25)