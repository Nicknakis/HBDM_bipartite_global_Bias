# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 13:47:48 2020

@author: nnak
"""

# Import all the packages
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.optim as optim
import torch.nn.functional as f# create a dummy data 
import timeit
from fractal_main_bip import Tree_kmeans_recursion
from missing_data import Create_missing_data
#from kmeans_cuda import Normal_Kmeans as Euclidean_Kmeans
from copy import deepcopy
# from blobs import *
CUDA = torch.cuda.is_available()
from spectral_clustering import Spectral_clustering_init
from sklearn import metrics
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import seaborn as sns
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if device.type=='cuda':
    print('Running on GPU')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    print('Running on CPU (slow)')
    torch.set_default_tensor_type('torch.FloatTensor')

    
    
undirected=1
from scipy.sparse import coo_matrix,csr_matrix

import matplotlib.pyplot as plt


class LSM(nn.Module,Tree_kmeans_recursion,Create_missing_data,Spectral_clustering_init):
    def __init__(self,link_function,sparse_i,sparse_j, input_size_1,input_size_2,latent_dim,non_sparse_i=None,non_sparse_j=None,sparse_i_rem=None,sparse_j_rem=None,CVflag=True,graph_type='undirected',missing_data=False):
        super(LSM, self).__init__()
        Tree_kmeans_recursion.__init__(self,minimum_points=int(input_size_1/(input_size_1/np.log(input_size_1))),init_layer_split=torch.round(torch.log(torch.tensor(input_size_1).float())),device=device)
        Create_missing_data.__init__(self,percentage=0.2)
        Spectral_clustering_init.__init__(self,num_of_eig=latent_dim)
        self.input_size_1=input_size_1
        self.input_size_2=input_size_2
        self.latent_dim=latent_dim
        self.device=device
        
        self.link_function=link_function
       
        # Initialize latent space with the centroids provided from the Fractal over the spectral clustering space
        #self.kmeans_tree_recursively(depth=80,init_first_layer_cent=self.first_centers)
        self.bias=nn.Parameter(torch.randn(1,device=device))
        self.scaling_factor=nn.Parameter(torch.randn(1,device=device))
        self.softplus=nn.Softplus()
        
        
        self.graph_type=graph_type
        self.initialization=1
        self.scaling=1
        #create indices to index properly the receiver and senders variable
        self.sparse_i_idx=sparse_i
        self.flag1=0
        self.sparse_j_idx=sparse_j
        self.pdist = nn.PairwiseDistance(p=2,eps=0)
        self.missing_data=missing_data
        
        
        
        
        self.non_sparse_j_idx_removed=non_sparse_j
        self.non_sparse_i_idx_removed=non_sparse_i
           
        self.sparse_i_idx_removed=sparse_i_rem
        self.sparse_j_idx_removed=sparse_j_rem
        if sparse_i_rem!=None:
            self.removed_i=torch.cat((self.non_sparse_i_idx_removed,self.sparse_i_idx_removed))
            self.removed_j=torch.cat((self.non_sparse_j_idx_removed,self.sparse_j_idx_removed))
             
          
    
            
      
           
        self.first_centers=torch.randn(int(torch.round(torch.log(torch.tensor(input_size_1).float()))),latent_dim,device=device)
      
        spectral_centroids_to_z,spectral_centroids_to_w=self.spectral_clustering()
       
        self.latent_z=nn.Parameter(spectral_centroids_to_z)
        self.latent_w=nn.Parameter(spectral_centroids_to_w)
       
        # self.latent_z=nn.Parameter(torch.randn(self.input_size_1,self.latent_dim))
        # self.latent_w=nn.Parameter(torch.randn(self.input_size_2,self.latent_dim))
        # self.gamma=nn.Parameter(torch.randn(self.input_size_1,device=device))
       
        # self.alpha=nn.Parameter(torch.randn(self.input_size_2,device=device))
               
        # self.latent_z=nn.Parameter(torch.randn(self.input_size_1,self.latent_dim))
        # self.latent_w=nn.Parameter(torch.randn(self.input_size_2,self.latent_dim))
      


    def local_likelihood(self,analytical_i,analytical_j):
        '''

        Parameters
        ----------
        k_mask : data points belonging to the specific centroid

        Returns
        -------
        Explicit calculation over the box of a specific centroid

        '''
        #change the distance to matrix and then reuse the Z^T matrix to calculate everything
        #return
       
            
     
        block_pdist=self.pdist(self.latent_z[analytical_i],self.latent_w[analytical_j])+1e-08
                
        ## Block kmeans analytically#########################################################################################################
                
        lambda_block=-block_pdist+self.bias
        if self.link_function=='EXP':

            an_lik=torch.exp(lambda_block).sum()
        elif self.link_function=='SOFTPLUS':

            an_lik=self.softplus(lambda_block).sum()
            
        else:
            raise ValueError('Invalid link function choice')
        return an_lik
        
    
    #introduce the likelihood function containing the two extra biases gamma_i and alpha_j
    def LSM_likelihood_bias(self,epoch):
        '''

        Parameters
        ----------
        cent_dist : real
            distnces of the updated centroid and the k-1 other centers.
        count_prod : TYPE
            DESCRIPTION.
        mask : Boolean
            DESCRIBES the slice of the mask for the specific kmeans centroid.

        Returns
        -------
        None.

        '''
        self.epoch=epoch
        
          
  
        analytical_i,analytical_j,thetas,init_centroids=self.kmeans_tree_recursively(depth=80,initial_cntrs=self.first_centers)
        self.first_centers=init_centroids
        #theta_stack=torch.stack(self.thetas).sum()
        analytical_blocks_likelihood=self.local_likelihood(analytical_i,analytical_j)
        ##############################################################################################################################
        self.analytical_i=analytical_i
        self.analytical_j=analytical_j
        z_pdist=(((self.latent_z[self.sparse_i_idx]-self.latent_w[self.sparse_j_idx]+1e-06)**2).sum(-1))**0.5
                
        ####################################################################################################################################
                
                                
                #take the sampled matrix indices in order to index gamma_i and alpha_j correctly and in agreement with the previous
                #remember the indexing of the z_pdist vector
               
               
        logit_u=-z_pdist+self.bias
         #########################################################################################################################################################      
        log_likelihood_sparse=torch.sum(logit_u)-thetas-analytical_blocks_likelihood
        #############################################################################################################################################################        
                 
            
        return log_likelihood_sparse
    
    
    
    def link_prediction(self):
        with torch.no_grad():
            z_pdist_miss=(((self.latent_z[self.removed_i]-self.latent_w[self.removed_j])**2).sum(-1))**0.5
            logit_u_miss=-z_pdist_miss+self.bias
            
            if self.link_function=='EXP':

            
                rates=torch.exp(logit_u_miss)
                
            elif self.link_function=='SOFTPLUS':

            
                rates=self.softplus(logit_u_miss)
            else:
                raise ValueError('Invalid link function choice')
            
                
                
            self.rates=rates

        
            target=torch.cat((torch.zeros(self.non_sparse_i_idx_removed.shape[0]),torch.ones(self.sparse_i_idx_removed.shape[0])))
            precision, recall, thresholds = metrics.precision_recall_curve(target.cpu().data.numpy(), rates.cpu().data.numpy())

           
        return metrics.roc_auc_score(target.cpu().data.numpy(),rates.cpu().data.numpy()),metrics.auc(recall,precision)
    

    

    
torch.autograd.set_detect_anomaly(True)

latent_dims=[2]


# Available choices are ['EXP','SOFTPLUS']
link_function='EXP'

datasets=['drug_gene']
for dataset in datasets:
    for latent_dim in latent_dims:
        # N=4039
        #N=36692
        
        print(latent_dim)
        rocs=[]
        prs=[]
        for cv_split in range(1):
            print(dataset)
            losses=[]
            ROC=[]
            PR=[]
            zetas=[]
            betas=[]
            scalings=[]
           
    
    
    # ################################################################################################################################################################
    # ################################################################################################################################################################
    # ################################################################################################################################################################
            sparse_i_rem=torch.from_numpy(np.loadtxt('./'+dataset+'/sparse_i_rem.txt')).long().to(device)
            sparse_j_rem=torch.from_numpy(np.loadtxt('./'+dataset+'/sparse_j_rem.txt')).long().to(device)
            non_sparse_i=torch.from_numpy(np.loadtxt('./'+dataset+'/non_sparse_i.txt')).long().to(device)
            non_sparse_j=torch.from_numpy(np.loadtxt('./'+dataset+'/non_sparse_j.txt')).long().to(device)
            
            
            import pandas as pd

            sparse_i=pd.read_csv('./'+dataset+'/sparse_i.txt')
            sparse_i=torch.tensor(sparse_i.values.reshape(-1),device=device).long()
            
            sparse_j=pd.read_csv('./'+dataset+'/sparse_j.txt')
            sparse_j=torch.tensor(sparse_j.values.reshape(-1),device=device).long()


         
            N1=int(sparse_i.max()+1)
            N2=int(sparse_j.max()+1)

           
            model = LSM(link_function,sparse_i,sparse_j,N1,N2,latent_dim=latent_dim,non_sparse_i=non_sparse_i,non_sparse_j=non_sparse_j,sparse_i_rem=sparse_i_rem,sparse_j_rem=sparse_j_rem,CVflag=True,graph_type='undirected',missing_data=False).to(device)
    
            optimizer = optim.Adam(model.parameters(), 0.1)  
           
            model.scaling=0
            for epoch in range(10001):
                                  
                
                loss=-model.LSM_likelihood_bias(epoch=epoch)/N1
               
                
         
             
                optimizer.zero_grad() # clear the gradients.   
                loss.backward() # backpropagate
                optimizer.step() # update the weights
                # scheduler.step()
                if epoch%100==0:
                    print(loss.item())
                    print(epoch)

                   
                    roc,pr=model.link_prediction() #perfom link prediction and return auc-roc, auc-pr
                   
                    print(roc,pr)
                
            
            