import scipy, copy
from tqdm import tqdm
import seaborn as sns
from datetime import datetime

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

def initialize_worm_models(slds):
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



def train_model(n_disc_states, n_latent_dim, filestr):
    slds, q = train_global_rslds(n_disc_states, n_latent_dim)

    with open(filestr+"/saved_data/prior.npy", 'wb') as handle:
        pickle.dump(slds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(filestr+"/saved_data/prior_q.npy", 'wb') as handle:   
        pickle.dump(q, handle, protocol=pickle.HIGHEST_PROTOCOL)

    global_prior, worm_models = initialize_worm_models(slds)

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
