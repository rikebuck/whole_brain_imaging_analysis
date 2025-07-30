import ssm
from ssm.util import random_rotation, find_permutation

import os
import numpy as np
import matplotlib.pyplot as plt

import pickle
import pickle


import sys

import scipy, copy
from tqdm import tqdm
import seaborn as sns

sys.path.append("/Users/friederikebuck/")

sys.path.append("/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/")
sys.path.append('/Users/bennetsakelaris/Documents/Obsidian Vault/Worms/wormcode/Code+Notes 09-24-24/collab/')


# from rslds_utils.load_data_utils import load_all_data_but_pretend_its_all_one_worm, load_all_data
# from rslds_utils.rslds_plotting_utils import *
from rslds_utils.rslds_eval_plotting_utils import plot_and_save, plot_var_explained, plot_and_save_vanilla
from rslds_utils.simulation_utils import inhibit_rim




# emissions_dim = neural_labels.shape[0]
def train_vanilla_rslds(Y,# emsions = neural acitivity
                       masks, # 1 where neuron exists 0 where nan 
                       z, # beh time series
                       tags, #not sure what this is..? 
                       n_disc_states, latent_dim,
                       emissions_dim,
                       T = 1599, 
                       i_want_to_plot_fitting = False, 
                       ):
    
    #rSLDS assumptions
    transition = "recurrent_only"
    #transition = "sticky_recurrent_only"
    dynamic = "diagonal_gaussian"
    emission = "gaussian_orthog"
    # Create the model and initialize its parameters
    slds = ssm.SLDS(emissions_dim, n_disc_states, latent_dim,
                    transitions=transition,
                    dynamics=dynamic, 
                    emissions=emission, 
                    single_subspace=True, verbose=False)


    # Fit the model using Laplace-EM with a structured variational posterior
    q_elbos, q = slds.fit(Y, method="laplace_em", #default
                                variational_posterior="structured_meanfield", #default
                                num_iters=50, alpha=0, masks=masks, 
                                # tags=tags, 
                                # num_init_restarts=15,
                                # verbose=False
                                )
    
    try:
        slds.permute(find_permutation(z[0:T], slds.most_likely_states(q.mean_continuous_states[0], Y[0])))
        # print(q.mean_continuous_states[0].shape)
        # print(Y[0].shape)
        print("found permutation!!!!")
    except:
        pass

    if i_want_to_plot_fitting:
        # plot results of SLDS fitting to make sure it converged
        fig, axs = plt.subplots(1, 1)
        axs.plot(q_elbos)
        axs.set_xlabel("Iteration")
        axs.set_ylabel("ELBO")
        plt.tight_layout()

    return slds,q_elbos,  q

def train_global_rslds(Y,# emsions = neural acitivity
                       masks, # 1 where neuron exists 0 where nan 
                       z, # beh time series
                       tags, #not sure what this is..? 
                       n_disc_states, latent_dim,
                       emissions_dim,
                       T = 1599, 
                       i_want_to_plot_fitting = False, 
                       ):
    
    #rSLDS assumptions
    transition = "recurrent_only"
    #transition = "sticky_recurrent_only"
    dynamic = "diagonal_gaussian"
    emission = "gaussian_orthog"
    # Create the model and initialize its parameters
    slds = ssm.SLDS(emissions_dim, n_disc_states, latent_dim,
                    transitions=transition,
                    dynamics=dynamic, 
                    emissions=emission, 
                    single_subspace=True, verbose=False)


    # Fit the model using Laplace-EM with a structured variational posterior
    q_elbos, q = slds.fit(Y, method="laplace_em", #default
                                variational_posterior="structured_meanfield", #default
                                num_iters=50, alpha=0, masks=masks, 
                                tags=tags, 
                                num_init_restarts=15,
                                verbose=False
                                )
    
    try:
        slds.permute(find_permutation(z[0:T], slds.most_likely_states(q.mean_continuous_states[0], Y[0])))
        print(q.mean_continuous_states[0].shape)
        print(Y[0].shape)
    except:
        pass

    if i_want_to_plot_fitting:
        # plot results of SLDS fitting to make sure it converged
        fig, axs = plt.subplots(1, 1)
        axs.plot(q_elbos)
        axs.set_xlabel("Iteration")
        axs.set_ylabel("ELBO")
        plt.tight_layout()

    return slds,q_elbos,  q

def initialize_worm_models(slds, Y):
    K = slds.K
    latent_dim = slds.D
    # Shared prior
    global_prior = {
        "A_mean": np.zeros((K, latent_dim, latent_dim)),# K is num discrete states  # this iwll be replaced by A_mean of all worms 
        "A_cov": np.eye(latent_dim * latent_dim)*0.04,
        "b_mean": np.zeros((K, latent_dim)), # this iwll be replaced by b_mean of all worms 
        "b_cov": np.eye(latent_dim)*.13,
        "Q_prior": {"Psi": np.eye(latent_dim), "nu": latent_dim + 2},# this iwll be replaced by mean Q_sum of all worms 
        "r_mean": np.zeros(K),  # will be updated by r mean .. review tha t r_mean is' and what R_mean is 
        "r_cov": np.eye(K),
        "R_mean": np.zeros((K, latent_dim)), # will be updated by r mean 
        "R_cov": np.eye(latent_dim)*.5,
    }

    # Worm-specific models
    worm_models = []
    for worm_data in Y:
        model = copy.deepcopy(slds)
        worm_models.append(model)
        
    return global_prior, worm_models


def initialize_global_prior(model):
    K = len(model.dynamics.As)  
    latent_dim = model.D  

    # Initialize global prior
    global_prior = {
        "A_mean": np.zeros((K, latent_dim, latent_dim)),
        "b_mean": np.zeros((K, latent_dim)),
        "Q_prior": {"Psi": np.eye(latent_dim), "nu": latent_dim + 2},
        "r_mean": np.zeros(K),
        "R_mean": np.zeros((K, latent_dim))
    }

    # Populate dynamics matrices A and biases b
    global_prior["A_mean"] = np.stack(model.dynamics.As, axis=0)
    global_prior["b_mean"] = np.stack(model.dynamics.bs, axis=0)
    
    # Populate noise covariance prior
    global_prior["Q_prior"]["Psi"] = np.mean(np.stack(model.dynamics.Sigmas, axis=0), axis=0)
    
    # Populate transition parameters
    global_prior["r_mean"] = model.transitions.r
    global_prior["R_mean"] = np.stack(model.transitions.Rs, axis=0)
    
    return global_prior



def update_global_prior(worm_models, global_prior):
    num_worms = len(worm_models)
    K = worm_models[0].K 
    
    # Initialize arrays 
    A_sum = np.zeros((K, worm_models[0].D, worm_models[0].D))
    b_sum = np.zeros((K, worm_models[0].D))
    Q_sum = np.zeros((K, worm_models[0].D, worm_models[0].D))
    r_sum = np.zeros(K)
    R_sum = np.zeros((K, worm_models[0].D))
    

     # Compute mean across worms
    for model in worm_models:
        A_sum += np.stack(model.dynamics.As, axis=0)  
        b_sum += np.stack(model.dynamics.bs, axis=0) 
        Q_sum += np.stack(model.dynamics.Sigmas, axis=0) 
        r_sum += model.transitions.r 
        R_sum += np.stack(model.transitions.Rs, axis=0)  
    
   
    global_prior["A_mean"] = A_sum / num_worms
    global_prior["b_mean"] = b_sum / num_worms
    global_prior["Q_prior"]["Psi"] = np.mean(Q_sum, axis=0)  # Update noise prior Psi # review

    global_prior["r_mean"] = r_sum / num_worms
    global_prior["R_mean"] = R_sum / num_worms
    
    return global_prior




def resample_parameters(model, global_prior):
    K = model.K

    model.dynamics.As = np.array([
        np.random.multivariate_normal(global_prior["A_mean"][k].flatten(), global_prior["A_cov"]).reshape(model.D, model.D) # so A_cov never updates? 
        for k in range(K)
    ])
    
    model.dynamics.bs = np.array([
        np.random.multivariate_normal(global_prior["b_mean"][k], global_prior["b_cov"])
        for k in range(K)
    ])

    model.dynamics.Sigmas = np.array([
        scipy.stats.invwishart.rvs(scale=global_prior["Q_prior"]["Psi"], df=global_prior["Q_prior"]["nu"])
        for k in range(K)
    ])

    model.transitions.r = np.random.multivariate_normal(global_prior["r_mean"], global_prior["r_cov"])
    model.transitions.Rs = np.array([
        np.random.multivariate_normal(global_prior["R_mean"][k], global_prior["R_cov"])
        for k in range(K)
    ])



def train_hierarchical_model(Y,# emsions = neural acitivity
                masks, # 1 where neuron exists 0 where nan 
                z, # beh time series
                tags, #not sure what this is..? 
                n_disc_states, latent_dim,
                emissions_dim,
                filestr,
                # T = 1599, 
                i_want_to_plot_fitting = False
                
                ):

    slds, q_elbos,  q= train_global_rslds(Y,# emsions = neural acitivity
                       masks, # 1 where neuron exists 0 where nan 
                       z, # beh time series
                       tags, #not sure what this is..? 
                       n_disc_states, latent_dim,
                       emissions_dim,
                       T = 1599, 
                       i_want_to_plot_fitting = i_want_to_plot_fitting)
    
    with open(filestr+"/saved_data/prior.npy", 'wb') as handle:
        pickle.dump(slds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(filestr+"/saved_data/prior_q.npy", 'wb') as handle:   
        pickle.dump(q, handle, protocol=pickle.HIGHEST_PROTOCOL)

    global_prior, worm_models = initialize_worm_models(slds, Y)

    # train each model
    
    for step in range(1):
            global_prior = update_global_prior(worm_models, global_prior)
            for model in worm_models:
                resample_parameters(model, global_prior)
            qs = []
            for worm, model in tqdm(enumerate(worm_models)):
                q_elbos, q_worm = model.fit(Y[worm], method="laplace_em", #default
                                        variational_posterior="structured_meanfield", #default
                                        num_iters=50, alpha=0, masks=[masks[worm]], initialize=False, verbose=False)
                qs.append(q_worm)
            global_prior = update_global_prior(worm_models, global_prior)

    with open(filestr+"/saved_data/worm_models.npy", 'wb') as handle:
        pickle.dump(worm_models, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(filestr+"/saved_data/q_data.npy", 'wb') as handle:
        pickle.dump(qs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return slds, qs, global_prior, worm_models





def train_and_eval_hierarchical_model(Y, z,tags, masks, neural_labels, emissions_dim, n_disc_states,latent_dim , 
                                       transition, dynamic, emission,  palette, cmap, formatted_datetime, save_dir = "", filestr_supp = "_hierch"
                                       ):
    np.random.seed(0)
    
    
    
   
    # now = datetime.now()
    # formatted_datetime = now.strftime("%Y%m%d_%H%M%S")
    # os.mkdir(formatted_datetime)
    # np.save(formatted_datetime+"/neurons.npy", neural_labels)
    # with open(formatted_datetime+"/Y.npy", 'wb') as handle:
    #     pickle.dump(Y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # Ds = np.arange(3,7,1)
    # Ds = np.arange(4,7,1)
    # # Ks = np.arange(3,4,1)
    # Ks = np.arange(4,5,1)
    # fitting_results = np.zeros((Ds.size,Ks.size))

    # i=0
    # for n_disc_states in Ks:
        # j=0
        # for latent_dim in Ds:
        
        
        
    # now = datetime.now()
    # formatted_datetime = now.strftime("%Y%m%d_%H%M%S")
    # # os.mkdir(formatted_datetime)
    # filestr = os.path.join(save_dir,  formatted_datetime)
    # os.mkdir(filestr, exist_ok = True)
    # np.save( os.path.join(save_dir, formatted_datetime, "neurons.npy"), neural_labels)
    # with open(os.path.join(save_dir, formatted_datetime, "Y.npy"), 'wb') as handle:
    #     pickle.dump(Y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # Ds = np.arange(3,7,1)
    # Ds = np.arange(4,7,1)
    # # Ks = np.arange(3,4,1)
    # Ks = np.arange(4,5,1)
    # fitting_results = np.zeros((Ds.size,Ks.size))

    # i=0
    # for n_disc_states in Ks:
        # j=0
        # for latent_dim in Ds:


    print(f"K = {n_disc_states}, D = {latent_dim}")
    filestr = os.path.join(save_dir,  formatted_datetime+f"/{filestr_supp}_N{emissions_dim}_D{latent_dim}_K{n_disc_states}")

    os.makedirs(filestr, exist_ok = True)
    os.makedirs(filestr+"/saved_data", exist_ok = True)
    os.makedirs(filestr+"/saved_figs",  exist_ok = True)


    slds, qs, global_prior, worm_models = train_hierarchical_model(Y,# emsions = neural acitivity
                masks, # 1 where neuron exists 0 where nan 
                z, # beh time series
                tags, #not sure what this is..? 
                n_disc_states, latent_dim,
                emissions_dim,
                filestr,
                # T = 1599, 
                i_want_to_plot_fitting = False
                
                )#train_model(n_disc_states, latent_dim, filestr)
    
    
    filename = os.path.join(filestr,"saved_data", "hier_ouputs.pickle")
    a = [worm_models, qs,Y, z, filestr,emissions_dim, transition, dynamic, emission]
    with open(filename, 'wb') as file:
        # Pickle the object and write it to the file
        pickle.dump(a, file)
    # q_x_full = plot_and_save(worm_models, qs, filestr)
    q_x_full = plot_and_save(worm_models, qs,Y, z, filestr,emissions_dim, transition, dynamic, emission,  palette, cmap, 
                  )
    
    # inhibit_rim(slds, q_x_full[11], filestr, "")
    inhibit_rim(slds, q_x_full[11], filestr, "", neural_labels)
    # var_explained = plot_var_explained(worm_models, q_x_full, filestr)
    var_explained = plot_var_explained(worm_models, q_x_full,neural_labels,  masks,Y, filestr, )
    # fitting_results[j][i] = var_explained
    # j+=1
        
        # fig,ax = plt.subplots()
        # ax.plot(Ds, fitting_results[:,i])
        # ax.set_xlabel("# latent dims")
        # ax.set_ylabel("var explained")
        # ax.set_ylim(0,1)
        # fig.savefig(formatted_datetime+f"/K{n_disc_states}.png")
        # plt.close()
        # i+=1
    # np.save(formatted_datetime+"/var_explained.npy",fitting_results)
    return var_explained


def train_and_eval_vanilla_model(Y, z,tags, masks, neural_labels, emissions_dim, n_disc_states, latent_dim, 
                                       transition, dynamic, emission,  palette, cmap, formatted_datetime,  save_dir = "", filestr_supp = "_vanilla", return_var_explained = True):
    np.random.seed(0)
    # now = datetime.now()
    # formatted_datetime = now.strftime("%Y%m%d_%H%M%S")
    # # os.mkdir(formatted_datetime)
    # filestr = os.path.join(save_dir,  formatted_datetime)
    # os.mkdir(filestr, exist_ok = True)
    # np.save( os.path.join(save_dir, formatted_datetime, "neurons.npy"), neural_labels)
    # with open(os.path.join(save_dir, formatted_datetime, "Y.npy"), 'wb') as handle:
    #     pickle.dump(Y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # Ds = np.arange(3,7,1)
    # Ds = np.arange(4,7,1)
    # # Ks = np.arange(3,4,1)
    # Ks = np.arange(4,5,1)
    # fitting_results = np.zeros((Ds.size,Ks.size))

    # i=0
    # for n_disc_states in Ks:
        # j=0
        # for latent_dim in Ds:
    print(f"K = {n_disc_states}, D = {latent_dim}")
    filestr = os.path.join(save_dir,  formatted_datetime+f"/{filestr_supp}_N{emissions_dim}_D{latent_dim}_K{n_disc_states}")

    os.makedirs(filestr, exist_ok = True)
    os.makedirs(filestr+"/saved_data", exist_ok = True)
    os.makedirs(filestr+"/saved_figs",  exist_ok = True)


    # slds, q_elbos,  q = train_global_rslds(Y,# emsions = neural acitivity
    #                    masks, # 1 where neuron exists 0 where nan 
    #                    z, # beh time series
    #                    tags, #not sure what this is..? 
    #                    n_disc_states, latent_dim,
    #                    emissions_dim,
    #                    T = 1599, 
    #                    i_want_to_plot_fitting = False)
    slds, q_elbos,  q = train_vanilla_rslds(Y,# emsions = neural acitivity
                       masks, # 1 where neuron exists 0 where nan 
                       z, # beh time series
                       tags, #not sure what this is..? 
                       n_disc_states, latent_dim,
                       emissions_dim,
                       T = 1599, 
                       i_want_to_plot_fitting = False, 
                       )
    
    worm_models = [slds]
    qs = [q]

    # q_x_full = plot_and_save(worm_models, qs, Y, z, filestr,emissions_dim, transition, dynamic, emission,  palette, cmap, 
    #               )

    filename = os.path.join(filestr,"saved_data", "hier_ouputs.pickle")
    a = [worm_models, qs,Y, z, filestr,emissions_dim, transition, dynamic, emission]
    with open(filename, 'wb') as file:
        # Pickle the object and write it to the file
        pickle.dump(a, file)
    
    q_x_full = plot_and_save_vanilla(worm_models, qs,Y, z, filestr,emissions_dim,latent_dim, transition, dynamic, emission,  palette, cmap, 
                  )
    inhibit_rim(slds, q_x_full[11], filestr, "", neural_labels)
    # if return_var_explained: 
    #     var_explained = plot_var_explained(worm_models, q_x_full,neural_labels,  masks,Y, filestr,)
    # else: 
    #     var_explained = None
    # fitting_results[j][i] = var_explained
    # j+=1
        
        # fig,ax = plt.subplots()
        # ax.plot(Ds, fitting_results[:,i])
        # ax.set_xlabel("# latent dims")
        # ax.set_ylabel("var explained")
        # ax.set_ylim(0,1)
        # fig.savefig(formatted_datetime+f"/K{n_disc_states}.png")
        # plt.close()
        # i+=1
    # np.save(formatted_datetime+"/var_explained.npy",fitting_results)
    # return var_explained, worm_models, q_x_full, filestr
    return worm_models, q_x_full, filestr





def train_and_eval_vanilla_model_1(Y, z,tags, masks, neural_labels, emissions_dim, n_disc_states, latent_dim, 
                                       transition, dynamic, emission,  palette, cmap, formatted_datetime,  save_dir = "", filestr_supp = "_vanilla"):
    # np.random.seed(0)

    print(f"K = {n_disc_states}, D = {latent_dim}")
    filestr = os.path.join(save_dir,  formatted_datetime+f"/{filestr_supp}_N{emissions_dim}_D{latent_dim}_K{n_disc_states}")

    os.makedirs(filestr, exist_ok = True)
    os.makedirs(filestr+"/saved_data", exist_ok = True)
    os.makedirs(filestr+"/saved_figs",  exist_ok = True)
    #rSLDS assumptions
    transition = "recurrent_only"
    #transition = "sticky_recurrent_only"
    dynamic = "diagonal_gaussian"
    emission = "gaussian_orthog"


    # if transition == "sticky_recurrent_only":
    #     transition_model = StickyRecurrentOnlyTransitions(K=n_disc_states,D=latent_dim, l2_penalty_similarity=10, l1_penalty=10) 

    #     # Create the model and initialize its parameters
    #     slds = ssm.SLDS(emissions_dim, n_disc_states, latent_dim,
    #                     transitions=transition_model,
    #                     dynamics=dynamic, 
    #                     emissions=emission,
    #                     single_subspace=True)
    # else: 
    # Create the model and initialize its parameters
    slds = ssm.SLDS(emissions_dim, n_disc_states, latent_dim,
                    transitions=transition,
                    dynamics=dynamic, 
                    emissions=emission,
                    single_subspace=True)


    # Fit the model using Laplace-EM with a structured variational posterior
    q_elbos, q = slds.fit(Y, method="laplace_em", #default
                                variational_posterior="structured_meanfield", #default
                                num_iters=50, alpha=0, masks=masks)
    slds.permute(find_permutation(z[0:1599], slds.most_likely_states(q.mean_continuous_states[0], Y[0])))

    i_want_to_plot_fitting = False
    if i_want_to_plot_fitting:
        # plot results of SLDS fitting to make sure it converged
        fig, axs = plt.subplots(1, 1)
        axs.plot(q_elbos)
        axs.set_xlabel("Iteration")
        axs.set_ylabel("ELBO")
        plt.tight_layout()

    # slds, q_elbos,  q = train_vanilla_rslds(Y,# emsions = neural acitivity
    #                    masks, # 1 where neuron exists 0 where nan 
    #                    z, # beh time series
    #                    tags, #not sure what this is..? 
    #                    n_disc_states, latent_dim,
    #                    emissions_dim,
    #                    T = 1599, 
    #                    i_want_to_plot_fitting = False, 
    #                    )
    
    worm_models = [slds]
    qs = [q]

    # q_x_full = plot_and_save(worm_models, qs, Y, z, filestr,emissions_dim, transition, dynamic, emission,  palette, cmap, 
    #               )
    
    q_x_full = plot_and_save_vanilla(worm_models, qs,Y, z, filestr,emissions_dim,latent_dim, transition, dynamic, emission,  palette, cmap, 
                )
    inhibit_rim(slds, q_x_full[11], filestr, "", neural_labels)
    var_explained = plot_var_explained(worm_models, q_x_full,neural_labels,  masks,Y, filestr,)
    # fitting_results[j][i] = var_explained
    # j+=1
        
        # fig,ax = plt.subplots()
        # ax.plot(Ds, fitting_results[:,i])
        # ax.set_xlabel("# latent dims")
        # ax.set_ylabel("var explained")
        # ax.set_ylim(0,1)
        # fig.savefig(formatted_datetime+f"/K{n_disc_states}.png")
        # plt.close()
        # i+=1
    # np.save(formatted_datetime+"/var_explained.npy",fitting_results)
    return var_explained