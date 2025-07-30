import ssm
from ssm.util import random_rotation, find_permutation

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

import sys

sys.path.append("/Users/friederikebuck/")

sys.path.append("/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/")
sys.path.append('/Users/bennetsakelaris/Documents/Obsidian Vault/Worms/wormcode/Code+Notes 09-24-24/collab/')

sys.path.append("/ru-auth/local/home/fbuck/scratch/test_rslds_params/")
from rslds_utils.load_data_utils import load_all_data_but_pretend_its_all_one_worm, load_all_data
from rslds_utils.rslds_plotting_utils import plot_2d_continuous_states, plot_most_likely_dynamics
from rslds_utils.subsample_neurons import *



def plot_and_save(worm_models, qs,Y, z, filestr,emissions_dim, transition, dynamic, emission,  palette, cmap, 
                  ):
    q_z_full = []
    q_x_full = []
    n_disc_states = worm_models[0].K

    # for w, model in enumerate(worm_models):
    for w in range(len(Y)): 
        if len(worm_models)>1:
            model = worm_models[w]
            q = qs[w]
        else: 
            model = worm_models[0]
            q = qs[0]
        
            
        # Get the posterior mean of the continuous states
        q_x = q.mean_continuous_states[0]
        Y_w = Y[w]
        
        z_w = z[w*1599:(w+1)*1599]
        q_z = model.most_likely_states(q_x, Y_w) #this is estimated behavioral state
        q_x_full.append(q_x)

        try:
            model.permute(find_permutation(z_w, q_z))
        except:
            pass
        
        q_z = model.most_likely_states(q_x, Y_w) #this is estimated behavioral state
        q_z_full.append(q_z)

        # Plot state overlap as percent of actual
        overlap = np.zeros((3, n_disc_states))
        denom = []
        for i in range(3):
            denom.append(sum(z_w==i))
        for t in range(len(z_w)):
            overlap[z_w[t],q_z[t]] +=1.0/denom[z_w[t]]
        # if True:
        fig, ax = plt.subplots(1,1, figsize=(8,6))
        im = ax.imshow(overlap, vmin=0, vmax=1)
        ax.set_xlabel("estimated")
        ax.set_ylabel("true")
        fig.colorbar(im, ax=ax)
        ax.set_yticks(np.arange(0,3), labels=["forwards", "backwards", "turning"], rotation=0)    
        ax.set_xticks(np.arange(0,n_disc_states))
        ax.set_title("State overlap, worm {}".format(w))
        
        for i in range(3):
            for j in range(n_disc_states):
                ax.text(j, i, f"{overlap[i, j]:.2f}", ha='center', va='center', color='white')# if overlap[i, j] < 0.5 else 'black')

        fig.savefig(filestr+"/saved_figs/"+f"worm{w}_overlap.png")
        plt.close()

        # Plot the true and inferred states
        fig, axs = plt.subplots(2,1, figsize=(18,6))
        axs[0].imshow(z_w[None,:], aspect="auto", cmap=cmap, alpha=0.3, vmin=0, vmax=len(palette))
        axs[1].imshow(q_z[None,:], aspect="auto", cmap=cmap, alpha=0.3, vmin=0, vmax=len(palette))
        axs[0].set_yticks([]); axs[1].set_yticks([])
        axs[0].set_title("Given labels"); axs[1].set_title("Inferred by rSLDS")
        axs[1].set_xticks([])
        fig.savefig(filestr+"/saved_figs/"+f"worm{w}_states.png")
        plt.close()
            


        #phase portraits
        pca = PCA() 
        pca_x = pca.fit_transform(q_x) #do pca
        id1=0 #choose which 2 PCs you want to plot
        id2=1
        W = pca.components_[[id1,id2],:]  # get components


        # Create the PCA-space rslds model and initialize its parameters
        # This is done so we can feed it into to scotts handy plot_most_likely_dynamics function
        pca_slds = ssm.SLDS(emissions_dim, n_disc_states, 2,
                        transitions=transition,
                        dynamics=dynamic,
                        emissions=emission,
                        single_subspace=True)


        #plot the phase portraits in PCA world
        for k in range(n_disc_states):
            # Dimensionality-reduced versions of A and b
            A_reduced = W @ model.dynamics.As[k] @ W.T
            b_reduced = W @ model.dynamics.bs[k]
            R_reduced = W @ model.transitions.Rs[k]
            pca_slds.dynamics.As[k] = A_reduced
            pca_slds.dynamics.bs[k] = b_reduced
            pca_slds.transitions.Rs[k] = R_reduced
            pca_slds.transitions.r[k] = model.transitions.r[k]

        if True:
            fig, ax = plt.subplots(1,1, figsize=(8,6))
            plot_2d_continuous_states(pca_x, q_z, ax=ax, inds=(id1, id2), lw=1, alpha=0.4)

            lim = abs(pca_x).max(axis=0) + 1
            try:
                pca_slds.permute(find_permutation(z_w, q_z))
            except:
                pass
            plot_most_likely_dynamics(pca_slds, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
            ax.set_title("PCs {} and {}, variance explained: {}".format(1+id1, 1+id2, sum(pca.explained_variance_ratio_[[id1,id2]])))


            plt.tight_layout()
            fig.savefig(filestr+"/saved_figs/"+f"worm{w}_pca.png")
            plt.close()
    return q_x_full

def variance_explained(y_pred, y_true):
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - sse / sst

def plot_var_explained(worm_models, q_x_full,neural_labels,  masks,Y, filestr, ):

    q_y = []
    y_hat = []

    for w in range(len(Y)): 
        if len(worm_models)>1:
            model = worm_models[w]
        else: 
            model = worm_models[0]
        y_hat.append(Y[w])
        q_y.append(model.smooth(q_x_full[w],Y[w]))
    var_explained = variance_explained(np.concatenate(q_y).flatten()[np.concatenate(masks).flatten()==1],
                                    np.concatenate(y_hat).flatten()[np.concatenate(masks).flatten()==1])

    fig,ax = plt.subplots(figsize=(16,6))

    neurons = np.zeros((neural_labels.size, len(q_y))) + np.nan
    for w in range(len(q_y)):
        for neuron in range(worm_models[0].N):
            if sum(Y[w][:,neuron]) !=0:
                neurons[neuron, w] = variance_explained(q_y[w][:,neuron], Y[w][:,neuron])
    sns.boxplot(neurons.T, ax=ax)
    np.save(filestr+"/neural_var_explained.npy", neurons)
    ax.set_xticks(np.arange(neural_labels.size), neural_labels, rotation=90, fontsize=4);
    ax.set_ylim(0, 1.1)
    ax.set_title(f"var explained:{var_explained}")
    fig.savefig(filestr+"/saved_figs/"+"var_explained.pdf")
    plt.close()

    return var_explained



# def plot_and_save_vanilla(worm_models, qs,Y, z, filestr,emissions_dim,latent_dim, transition, dynamic, emission,  palette, cmap, 
#                   ):
#     # q_z_full = []
#     # q_x_full = []
#     q_z_full = np.zeros(1599*len(Y))
#     q_x_full = np.zeros((1599*len(Y), latent_dim))
#     n_disc_states = worm_models[0].K

#     # for w, model in enumerate(worm_models):
#     for w in range(len(Y)): 
#         if len(worm_models)>1:
#             model = worm_models[w]
#             q = qs[w]
#         else: 
#             model = worm_models[0]
#             q = qs[0]
        
            
#         # Get the posterior mean of the continuous states
#         # q_x = q.mean_continuous_states[0]# <<<was this the issue..? seems like it.. 
#         q_x = q.mean_continuous_states[w]#this is estimated behavioral state <<<was this the issue..? seems like it.. 
#         Y_w = Y[w]
        
#         z_w = z[w*1599:(w+1)*1599]
#         q_x_full[w*1599:(w+1)*1599,:] = q_x
#         # q_z = model.most_likely_states(q_x, Y_w) #this is estimated behavioral state <<<was this the issue..? seems like it.. 
#         # q_x_full.append(q_x)

#         try:
#             model.permute(find_permutation(z_w, q_z))
#         except:
#             pass
        
#         q_z = model.most_likely_states(q_x, Y_w) #this is estimated behavioral state
#         # q_z_full.append(q_z)
#         q_z_full[w*1599:(w+1)*1599] =  q_z

#         # Plot state overlap as percent of actual
#         overlap = np.zeros((3, n_disc_states))
#         denom = []
#         for i in range(3):
#             denom.append(sum(z_w==i))
#         for t in range(len(z_w)):
#             overlap[z_w[t],q_z[t]] +=1.0/denom[z_w[t]]
#         # if True:
#         fig, ax = plt.subplots(1,1, figsize=(8,6))
#         im = ax.imshow(overlap, vmin=0, vmax=1)
#         ax.set_xlabel("estimated")
#         ax.set_ylabel("true")
#         fig.colorbar(im, ax=ax)
#         ax.set_yticks(np.arange(0,3), labels=["forwards", "backwards", "turning"], rotation=0)    
#         ax.set_xticks(np.arange(0,n_disc_states))
#         ax.set_title("State overlap, worm {}".format(w))
        
#         for i in range(3):
#             for j in range(n_disc_states):
#                 ax.text(j, i, f"{overlap[i, j]:.2f}", ha='center', va='center', color='white')# if overlap[i, j] < 0.5 else 'black')

#         fig.savefig(filestr+"/saved_figs/"+f"worm{w}_overlap.png")
#         plt.close()

#         # Plot the true and inferred states
#         fig, axs = plt.subplots(2,1, figsize=(18,6))
#         axs[0].imshow(z_w[None,:], aspect="auto", cmap=cmap, alpha=0.3, vmin=0, vmax=len(palette))
#         axs[1].imshow(q_z[None,:], aspect="auto", cmap=cmap, alpha=0.3, vmin=0, vmax=len(palette))
#         axs[0].set_yticks([]); axs[1].set_yticks([])
#         axs[0].set_title("Given labels"); axs[1].set_title("Inferred by rSLDS")
#         axs[1].set_xticks([])
#         fig.savefig(filestr+"/saved_figs/"+f"worm{w}_states.png")
#         plt.close()
            


#         #phase portraits
#         pca = PCA() 
#         pca_x = pca.fit_transform(q_x) #do pca
#         id1=0 #choose which 2 PCs you want to plot
#         id2=1
#         W = pca.components_[[id1,id2],:]  # get components


#         # Create the PCA-space rslds model and initialize its parameters
#         # This is done so we can feed it into to scotts handy plot_most_likely_dynamics function
#         pca_slds = ssm.SLDS(emissions_dim, n_disc_states, 2,
#                         transitions=transition,
#                         dynamics=dynamic,
#                         emissions=emission,
#                         single_subspace=True)


#         #plot the phase portraits in PCA world
#         for k in range(n_disc_states):
#             # Dimensionality-reduced versions of A and b
#             A_reduced = W @ model.dynamics.As[k] @ W.T
#             b_reduced = W @ model.dynamics.bs[k]
#             R_reduced = W @ model.transitions.Rs[k]
#             pca_slds.dynamics.As[k] = A_reduced
#             pca_slds.dynamics.bs[k] = b_reduced
#             pca_slds.transitions.Rs[k] = R_reduced
#             pca_slds.transitions.r[k] = model.transitions.r[k]

#         if True:
#             fig, ax = plt.subplots(1,1, figsize=(8,6))
#             plot_2d_continuous_states(pca_x, q_z, ax=ax, inds=(id1, id2), lw=1, alpha=0.4)

#             lim = abs(pca_x).max(axis=0) + 1
#             try:
#                 pca_slds.permute(find_permutation(z_w, q_z))
#             except:
#                 pass
#             plot_most_likely_dynamics(pca_slds, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
#             ax.set_title("PCs {} and {}, variance explained: {}".format(1+id1, 1+id2, sum(pca.explained_variance_ratio_[[id1,id2]])))


#             plt.tight_layout()
#             fig.savefig(filestr+"/saved_figs/"+f"worm{w}_pca.png")
#             plt.close()
#     return q_x_full


def plot_and_save_vanilla(worm_models, qs,Y, z, filestr,emissions_dim,latent_dim, transition, dynamic, emission,  palette, cmap, 
                  ):
    # q_z_full = []
    # q_x_full = []
    q_z_full = np.zeros(1599*len(Y))
    q_x_full = np.zeros((1599*len(Y), latent_dim))
    n_disc_states = worm_models[0].K

    # for w, model in enumerate(worm_models):
    for w in range(len(Y)): 
        if len(worm_models)>1:
            model = worm_models[w]
            q = qs[w]
        else: 
            model = worm_models[0]
            q = qs[0]
        
            
        # Get the posterior mean of the continuous states
        # q_x = q.mean_continuous_states[0]# <<<was this the issue..? seems like it.. 
        q_x = q.mean_continuous_states[w]#this is estimated behavioral state <<<was this the issue..? seems like it.. 
        Y_w = Y[w]
        
        z_w = z[w*1599:(w+1)*1599]
        q_x_full[w*1599:(w+1)*1599,:] = q_x
        # q_z = model.most_likely_states(q_x, Y_w) #this is estimated behavioral state <<<was this the issue..? seems like it.. 
        # q_x_full.append(q_x)
        q_z = model.most_likely_states(q_x, Y_w) #this is estimated behavioral state
        try:
            model.permute(find_permutation(z_w, q_z))
        except:
            pass
        
        
        # q_z_full.append(q_z)
        q_z_full[w*1599:(w+1)*1599] =  q_z

        # Plot state overlap as percent of actual
        overlap = np.zeros((3, n_disc_states))
        denom = []
        for i in range(3):
            denom.append(sum(z_w==i))
        for t in range(len(z_w)):
            overlap[z_w[t],q_z[t]] +=1.0/denom[z_w[t]]
        # if True:
        fig, ax = plt.subplots(1,1, figsize=(8,6))
        im = ax.imshow(overlap, vmin=0, vmax=1)
        ax.set_xlabel("estimated")
        ax.set_ylabel("true")
        fig.colorbar(im, ax=ax)
        ax.set_yticks(np.arange(0,3), labels=["forwards", "backwards", "turning"], rotation=0)    
        ax.set_xticks(np.arange(0,n_disc_states))
        ax.set_title("State overlap, worm {}".format(w))
        
        for i in range(3):
            for j in range(n_disc_states):
                ax.text(j, i, f"{overlap[i, j]:.2f}", ha='center', va='center', color='white')# if overlap[i, j] < 0.5 else 'black')

        fig.savefig(filestr+"/saved_figs/"+f"worm{w}_overlap.png")
        plt.close()

        # Plot the true and inferred states
        fig, axs = plt.subplots(2,1, figsize=(18,6))
        axs[0].imshow(z_w[None,:], aspect="auto", cmap=cmap, alpha=0.3, vmin=0, vmax=len(palette))
        axs[1].imshow(q_z[None,:], aspect="auto", cmap=cmap, alpha=0.3, vmin=0, vmax=len(palette))
        axs[0].set_yticks([]); axs[1].set_yticks([])
        axs[0].set_title("Given labels"); axs[1].set_title("Inferred by rSLDS")
        axs[1].set_xticks([])
        fig.savefig(filestr+"/saved_figs/"+f"worm{w}_states.png")
        plt.close()
            


        #phase portraits
        pca = PCA() 
        pca_x = pca.fit_transform(q_x) #do pca
        id1=0 #choose which 2 PCs you want to plot
        id2=1
        W = pca.components_[[id1,id2],:]  # get components


        # Create the PCA-space rslds model and initialize its parameters
        # This is done so we can feed it into to scotts handy plot_most_likely_dynamics function
        pca_slds = ssm.SLDS(emissions_dim, n_disc_states, 2,
                        transitions=transition,
                        dynamics=dynamic,
                        emissions=emission,
                        single_subspace=True)


        #plot the phase portraits in PCA world
        for k in range(n_disc_states):
            # Dimensionality-reduced versions of A and b
            A_reduced = W @ model.dynamics.As[k] @ W.T
            b_reduced = W @ model.dynamics.bs[k]
            R_reduced = W @ model.transitions.Rs[k]
            pca_slds.dynamics.As[k] = A_reduced
            pca_slds.dynamics.bs[k] = b_reduced
            pca_slds.transitions.Rs[k] = R_reduced
            pca_slds.transitions.r[k] = model.transitions.r[k]

        if True:
            fig, ax = plt.subplots(1,1, figsize=(8,6))
            plot_2d_continuous_states(pca_x, q_z, ax=ax, inds=(id1, id2), lw=1, alpha=0.4)

            lim = abs(pca_x).max(axis=0) + 1
            try:
                pca_slds.permute(find_permutation(z_w, q_z))
            except:
                pass
            plot_most_likely_dynamics(pca_slds, xlim=(-lim[0], lim[0]), ylim=(-lim[1], lim[1]), ax=ax)
            ax.set_title("PCs {} and {}, variance explained: {}".format(1+id1, 1+id2, sum(pca.explained_variance_ratio_[[id1,id2]])))


            plt.tight_layout()
            fig.savefig(filestr+"/saved_figs/"+f"worm{w}_pca.png")
            plt.close()
    return q_x_full