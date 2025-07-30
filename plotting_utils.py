
import os
import glob
import copy
import sys
import pickle 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np 
from scipy.stats import ks_2samp

def plot_probabilities_hist(np_array, bins, color = "blue", 
                            alpha = 0.7, label ="",
                             fig = None, ax = None, 
                             text ="", x_ratio = 9/12, y_ratio = 10/12, ylim = None, fill = True, 
                             linewidth=2):#fromstackoverflowhttps://stackoverflow.com/questions/3866520/plot-a-histogram-such-that-bar-heights-sum-to-1-probability
    if fig is None: 
        fig, ax = plt.subplots()
    weights = np.ones_like(np_array)/len(np_array)
    if not fill: 
       
        
        ax.hist(np_array, bins = bins, weights= weights, color = color, alpha = alpha, histtype='step',  label = label, fill = fill, linewidth = 2)
    else: 
         ax.hist(np_array, bins = bins, weights= weights, color = color, alpha = alpha, label = label)
    # if len(text)>0:
    #     add_text_to_upper_right_corner(ax, text,  x_ratio = x_ratio, y_ratio = y_ratio)
    # if ylim is None: 
    #     ax.set_ylim([0,1.1])
    return fig, ax


def plot_hist_across_conditins_and_get_KS_stats(cond_to_data,  conditon_to_color,
                                               title = "", 
                                               ylabel = "",
                                               xlabel = "",
                                               stat_color = "dark" ,
                                               bins = None, 
                                            #    ylim = [0,1.1],
                                            ylim = None, 
                                               alpha = 0.7, 
                                            #    linewidth=2, 
                                            fill =  False,
                                               fig = None, 
                                               ax = None , 
                                               print_sig = True,
                                               print_effect_size = False, 
                                               legend = False, condition_to_alpha = None, 
                                               plot_medians = False, 
                                               print_median_diff = False
                                
                                               ):
    
    
        
    if ax is None: 
        fig, ax = plt.subplots()
    medians = []
    for condition, data  in cond_to_data.items():
            
        counts, bin_edges = np.histogram(data, bins=bins, weights=np.ones_like(data)/len(data), density=False)
        cdf = np.cumsum(counts)
        
        # Normalize the CDF to ensure it ends at 1
        cdf = cdf / cdf[-1]

        if condition_to_alpha is None: 
            
            plot_probabilities_hist(data,  
                    bins,
                    conditon_to_color[condition], 
                    alpha = alpha, 
                    label = condition,
            
                    fig = fig, ax = ax, 
                    fill = fill, 
                    # text ="",
                    )
            
                
        else: 
            plot_probabilities_hist(data,  
                    bins,
                    conditon_to_color[condition], 
                    alpha = condition_to_alpha[condition], 
                    label = condition,
            
                    fig = fig, ax = ax, 
                    fill = fill, 
                    # text ="", 
                    )
        if plot_medians: 
            median = np.median(data)
            medians.append(median)
        
            ax.axvline(median, linestyle = "--", color = conditon_to_color[condition], alpha = alpha)
    if legend: 
        ax.legend()
    if ylim is not None: 
        ax.set_ylim(ylim)
    ax.set_title(title,  fontsize=19)
    ax.set_ylabel(ylabel, fontsize=19)
    ax.set_xlabel(xlabel, fontsize=19)


    ax.tick_params(axis='x', labelsize=14)  
    ax.tick_params(axis='y', labelsize=14)  
    
    # ax.spines['top'].set_visible(False)   # Hide top spine
    # ax.spines['right'].set_visible(False)  # Hide right spine

    # ax.spines['left'].set_linewidth(2)   # Left spine bold
    # ax.spines['bottom'].set_linewidth(2)  # Bottom spine bold
    # fig.set_size_inches(4, 5)

    # Compute KS Test if reference_data is provided
    ks_stat, p_value = None, None
    conditions = list(cond_to_data.keys())

    ks_stat, p_value = ks_2samp(cond_to_data[conditions[0]], cond_to_data[conditions[1]])  # 2-sample KS test
    
    # Determine significance stars
    if p_value < 0.001:
        stars = "***"
    elif p_value < 0.01:
        stars = "**"
    elif p_value < 0.05:
        stars = "*"
        
    else:
        stars = "ns"  # Not significant
    if print_effect_size and stars!= "ns": 
        stars = stars+ " " + str(np.round(ks_stat,2))
    if print_median_diff and len(medians)==2: 
        stars = stars+ " " + str(np.abs(np.round(np.diff(np.array(medians)),2))[0])
        ax.text(0.95, 0.95, stars, transform=ax.transAxes, fontsize=25, color=stat_color, ha='right', va='top' , fontweight='bold')
    
    if print_sig: 
    # Add stars to the plot in the lower-right corner
        ax.text(0.95, 0.95, stars, transform=ax.transAxes, fontsize=25, color=stat_color, ha='right', va='top' , fontweight='bold')

    
    # Optionally print results
    # print(f"KS Statistic: {ks_stat}, p-value: {p_value}")
    fig.tight_layout()
    return fig, ax 