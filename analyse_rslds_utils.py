import copy
import socket
import os
import pickle 
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt


sys.path.append("/Users/friederikebuck/Desktop/MBL/project/WholeBrainImagingAnalysis/collab/")
sys.path.append('/Users/bennetsakelaris/Documents/Obsidian Vault/Worms/wormcode/Code+Notes 09-24-24/collab/')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score
from sklearn.decomposition import PCA

def input_slds(slds, neural_labels, input_list_dynamics, input_list_emissions=[]):
    new_slds = copy.deepcopy(slds)

    new_slds.M += len(input_list_dynamics) + len(input_list_emissions)
    new_slds.emissions.M = new_slds.M
    new_slds.dynamics.M = new_slds.M
    new_slds.transitions.M = new_slds.M

    # Inititalize new input-driven properties
    Vs = np.zeros((new_slds.K, new_slds.D, new_slds.M))
    Fs = np.zeros((1, new_slds.N, new_slds.M))
    Ws = np.zeros((new_slds.K, new_slds.M))

    if slds.M > 0: #if old slds already supported inputs
        Vs[:,:,:slds.M] = slds.dynamics.Vs
        Ws[:,:slds.M] = slds.transitions.Ws
        Fs[:,:,:slds.M] = slds.emissions.Fs

    # Here I'm only changing Vs but maybe its worth changing Ws at some point
    input_dict = {} # neuron name -> input index
    for i in range(len(input_list_dynamics)):
        input_dict[input_list_dynamics[i]] = slds.M+i
        idx = np.argwhere(neural_labels == f"dF - {input_list_dynamics[i]}")[0][0]
        latent_vec = new_slds.emissions.Cs[0][idx]
        for k in range(new_slds.K):
            Vs[k, :,slds.M+i] = latent_vec

    for i in range(len(input_list_emissions)):
        input_dict["e"+input_list_emissions[i]] = slds.M + i + len(input_list_dynamics)
   
        idx = np.argwhere(neural_labels == f"dF - {input_list_emissions[i]}")[0][0]
        Fs[0,idx,slds.M+i+ len(input_list_dynamics)] = 10
        idx = np.argwhere(neural_labels == f"F - {input_list_emissions[i]}")[0][0]
        Fs[0,idx,slds.M+i+ len(input_list_dynamics)] = 10
                
    new_slds.dynamics.Vs = Vs
    new_slds.transitions.Ws = Ws
    new_slds.emissions.Fs = Fs
    return new_slds, input_dict

def make_behavior_decoder(X, y, test_size=0.2, random_state=42, quiet=True):

    if type(X) == list:    
        X = np.concatenate(X)
        y = np.concatenate(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Train logistic regression on the training data
    model = LogisticRegression(max_iter=1000, random_state=random_state, class_weight="balanced")
    model.fit(X_train, y_train)

    if quiet:
        return model

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)

    # Display the confusion matrix
    disp.plot(cmap='cividis')
    plt.title("Confusion Matrix -- decoding behavior from neural activity")
    plt.show()

    print("Balanced accuracy: ", balanced_accuracy_score(y_test, y_pred))
    return model
