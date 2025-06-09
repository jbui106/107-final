"""
Signal Detection Theory (SDT) and Delta Plot Analysis for Response Time Data
"""

import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import os

# Mapping dictionaries for categorical variables
# These convert categorical labels to numeric codes for analysis
MAPPINGS = {
    'stimulus_type': {'simple': 0, 'complex': 1},
    'difficulty': {'easy': 0, 'hard': 1},
    'signal': {'present': 0, 'absent': 1}
}

# Descriptive names for each experimental condition
CONDITION_NAMES = {
    0: 'Easy Simple',
    1: 'Easy Complex',
    2: 'Hard Simple',
    3: 'Hard Complex'
}

# Percentiles used for delta plot analysis
PERCENTILES = [10, 30, 50, 70, 90]

def read_data(file_path, prepare_for='sdt', display=False):
    """Read and preprocess data from a CSV file into SDT format.
    
    Args:
        file_path: Path to the CSV file containing raw response data
        prepare_for: Type of analysis to prepare data for ('sdt' or 'delta plots')
        display: Whether to print summary statistics
        
    Returns:
        DataFrame with processed data in the requested format
    """
    # Read and preprocess data
    data = pd.read_csv(file_path)
    
    # Convert categorical variables to numeric codes
    for col, mapping in MAPPINGS.items():
        data[col] = data[col].map(mapping)
    
    # Create participant number and condition index
    data['pnum'] = data['participant_id']
    data['condition'] = data['stimulus_type'] + data['difficulty'] * 2
    data['accuracy'] = data['accuracy'].astype(int)
    
    if display:
        print("\nRaw data sample:")
        print(data.head())
        print("\nUnique conditions:", data['condition'].unique())
        print("Signal values:", data['signal'].unique())
    
    # Transform to SDT format if requested
    if prepare_for == 'sdt':
        # Group data by participant, condition, and signal presence
        grouped = data.groupby(['pnum', 'condition', 'signal']).agg({
            'accuracy': ['count', 'sum']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['pnum', 'condition', 'signal', 'nTrials', 'correct']
        
        if display:
            print("\nGrouped data:")
            print(grouped.head())
        
        # Transform into SDT format (hits, misses, false alarms, correct rejections)
        sdt_data = []
        for pnum in grouped['pnum'].unique():
            p_data = grouped[grouped['pnum'] == pnum]
            for condition in p_data['condition'].unique():
                c_data = p_data[p_data['condition'] == condition]
                
                # Get signal and noise trials
                signal_trials = c_data[c_data['signal'] == 0]
                noise_trials = c_data[c_data['signal'] == 1]
                
                if not signal_trials.empty and not noise_trials.empty:
                    sdt_data.append({
                        'pnum': pnum,
                        'condition': condition,
                        'hits': signal_trials['correct'].iloc[0],
                        'misses': signal_trials['nTrials'].iloc[0] - signal_trials['correct'].iloc[0],
                        'false_alarms': noise_trials['nTrials'].iloc[0] - noise_trials['correct'].iloc[0],
                        'correct_rejections': noise_trials['correct'].iloc[0],
                        'nSignal': signal_trials['nTrials'].iloc[0],
                        'nNoise': noise_trials['nTrials'].iloc[0]
                    })
        
        data = pd.DataFrame(sdt_data)
        
        if display:
            print("\nSDT summary:")
            print(data)
            if data.empty:
                print("\nWARNING: Empty SDT summary generated!")
                print("Number of participants:", len(data['pnum'].unique()))
                print("Number of conditions:", len(data['condition'].unique()))
            else:
                print("\nSummary statistics:")
                print(data.groupby('condition').agg({
                    'hits': 'sum',
                    'misses': 'sum',
                    'false_alarms': 'sum',
                    'correct_rejections': 'sum',
                    'nSignal': 'sum',
                    'nNoise': 'sum'
                }).round(2))
    
    # Prepare data for delta plot analysis
    if prepare_for == 'delta plots':
        # Initialize DataFrame for delta plot data
        dp_data = pd.DataFrame(columns=['pnum', 'condition', 'mode', 
                                      *[f'p{p}' for p in PERCENTILES]])
        
        # Process data for each participant and condition
        for pnum in data['pnum'].unique():
            for condition in data['condition'].unique():
                # Get data for this participant and condition
                c_data = data[(data['pnum'] == pnum) & (data['condition'] == condition)]
                
                # Calculate percentiles for overall RTs
                overall_rt = c_data['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['overall'],
                    **{f'p{p}': [np.percentile(overall_rt, p)] for p in PERCENTILES}
                })])
                
                # Calculate percentiles for accurate responses
                accurate_rt = c_data[c_data['accuracy'] == 1]['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['accurate'],
                    **{f'p{p}': [np.percentile(accurate_rt, p)] for p in PERCENTILES}
                })])
                
                # Calculate percentiles for error responses
                error_rt = c_data[c_data['accuracy'] == 0]['rt']
                dp_data = pd.concat([dp_data, pd.DataFrame({
                    'pnum': [pnum],
                    'condition': [condition],
                    'mode': ['error'],
                    **{f'p{p}': [np.percentile(error_rt, p)] for p in PERCENTILES}
                })])
                
        if display:
            print("\nDelta plots data:")
            print(dp_data)
            
        data = pd.DataFrame(dp_data)

    return data


def apply_hierarchical_sdt_model(data):
    """Apply a hierarchical Signal Detection Theory model using PyMC.
    
    This function implements a Bayesian hierarchical model for SDT analysis,
    allowing for both group-level and individual-level parameter estimation.
    It includes stimulus type and difficulty as predictors.
    
    Args:
        data: DataFrame containing SDT summary statistics
        
    Returns:
        PyMC model object
    """
    # Get unique participants and conditions
    unique_pnums = data['pnum'].unique()
    P = len(unique_pnums)
    unique_conditions = data['condition'].unique()
    C = len(unique_conditions)
    
    stimulus_type_conditions = np.array([c % 2 for c in unique_conditions]) 
    difficulty_conditions = np.array([c // 2 for c in unique_conditions])  

    with pm.Model() as sdt_model:
        # Group-level parameters for effects of stimulus type and difficulty
        
        # d_prime effects
        mean_d_prime_intercept = pm.Normal('mean_d_prime_intercept', mu=0.0, sigma=1.0)
        effect_stimulus_type_dprime = pm.Normal('effect_stimulus_type_dprime', mu=0.0, sigma=0.5)
        effect_difficulty_dprime = pm.Normal('effect_difficulty_dprime', mu=0.0, sigma=0.5)
        stdev_d_prime_overall = pm.HalfNormal('stdev_d_prime_overall', sigma=1.0) # Overall SD for d_prime
        
        # criterion effects
        mean_criterion_intercept = pm.Normal('mean_criterion_intercept', mu=0.0, sigma=1.0)
        effect_stimulus_type_criterion = pm.Normal('effect_stimulus_type_criterion', mu=0.0, sigma=0.5)
        effect_difficulty_criterion = pm.Normal('effect_difficulty_criterion', mu=0.0, sigma=0.5)
        stdev_criterion_overall = pm.HalfNormal('stdev_criterion_overall', sigma=1.0) # Overall SD for criterion

        # Define the mean d_prime and criterion for *each condition (C)*
        mean_d_prime_per_condition = pm.Deterministic(
            'mean_d_prime_per_condition',
            mean_d_prime_intercept +
            effect_stimulus_type_dprime * stimulus_type_conditions +
            effect_difficulty_dprime * difficulty_conditions
        )
        
        mean_criterion_per_condition = pm.Deterministic(
            'mean_criterion_per_condition',
            mean_criterion_intercept +
            effect_stimulus_type_criterion * stimulus_type_conditions +
            effect_difficulty_criterion * difficulty_conditions
        )
        
        # Individual-level parameters
        d_prime = pm.Normal('d_prime',
                            mu=mean_d_prime_per_condition[data['condition'].values], # Index by condition
                            sigma=stdev_d_prime_overall,
                            shape=(P, C)) # This shape is still correct for the array of individual values

        # criterion for each participant and condition (P, C)
        criterion = pm.Normal('criterion',
                             mu=mean_criterion_per_condition[data['condition'].values], # Index by condition
                             sigma=stdev_criterion_overall,
                             shape=(P, C))
        
        # Calculate hit and false alarm rates using SDT
        hit_rate = pm.math.invlogit(d_prime[data['pnum'].values-1, data['condition'].values] - criterion[data['pnum'].values-1, data['condition'].values])
        false_alarm_rate = pm.math.invlogit(-criterion[data['pnum'].values-1, data['condition'].values])
                
        # Likelihood for signal trials
        pm.Binomial('hit_obs', 
                   n=data['nSignal'].values, # Use .values for observed data
                   p=hit_rate, 
                   observed=data['hits'].values)
        
        # Likelihood for noise trials
        pm.Binomial('false_alarm_obs', 
                   n=data['nNoise'].values, # Use .values for observed data
                   p=false_alarm_rate, 
                   observed=data['false_alarms'].values)
    
    return sdt_model


def draw_delta_plots(data, pnum):
    """Draw delta plots comparing RT distributions between condition pairs.
    
    Creates a matrix of delta plots where:
    - Upper triangle shows overall RT distribution differences
    - Lower triangle shows RT differences split by correct/error responses
    
    Args:
        data: DataFrame with RT percentile data
        pnum: Participant number to plot
    """
    # Filter data for specified participant
    data = data[data['pnum'] == pnum]
    
    # Get unique conditions and create subplot matrix
    conditions = data['condition'].unique()
    n_conditions = len(conditions)
    
    # Create figure with subplots matrix
    fig, axes = plt.subplots(n_conditions, n_conditions, 
                            figsize=(4*n_conditions, 4*n_conditions))
    
    # Create output directory
    OUTPUT_DIR = Path(__file__).parent.parent.parent / 'output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define marker style for plots
    marker_style = {
        'marker': 'o',
        'markersize': 10,
        'markerfacecolor': 'white',
        'markeredgewidth': 2,
        'linewidth': 3
    }
    
    # Create delta plots for each condition pair
    for i, cond1 in enumerate(conditions):
        for j, cond2 in enumerate(conditions):
            # Add labels only to edge subplots
            if j == 0:
                axes[i,j].set_ylabel('Difference in RT (s)', fontsize=12)
            if i == len(axes)-1:
                axes[i,j].set_xlabel('Percentile', fontsize=12)
                
            # Skip diagonal and lower triangle for overall plots
            if i > j:
                continue
            if i == j:
                axes[i,j].axis('off')
                continue
            
            # Create masks for condition and plotting mode
            cmask1 = data['condition'] == cond1
            cmask2 = data['condition'] == cond2
            overall_mask = data['mode'] == 'overall'
            error_mask = data['mode'] == 'error'
            accurate_mask = data['mode'] == 'accurate'
            
            # Calculate RT differences for overall performance
            quantiles1 = [data[cmask1 & overall_mask][f'p{p}'] for p in PERCENTILES]
            quantiles2 = [data[cmask2 & overall_mask][f'p{p}'] for p in PERCENTILES]
            overall_delta = np.array(quantiles2) - np.array(quantiles1)
            
            # Calculate RT differences for error responses
            error_quantiles1 = [data[cmask1 & error_mask][f'p{p}'] for p in PERCENTILES]
            error_quantiles2 = [data[cmask2 & error_mask][f'p{p}'] for p in PERCENTILES]
            error_delta = np.array(error_quantiles2) - np.array(error_quantiles1)
            
            # Calculate RT differences for accurate responses
            accurate_quantiles1 = [data[cmask1 & accurate_mask][f'p{p}'] for p in PERCENTILES]
            accurate_quantiles2 = [data[cmask2 & accurate_mask][f'p{p}'] for p in PERCENTILES]
            accurate_delta = np.array(accurate_quantiles2) - np.array(accurate_quantiles1)
            
            # Plot overall RT differences
            axes[i,j].plot(PERCENTILES, overall_delta, color='black', **marker_style)
            
            # Plot error and accurate RT differences
            axes[j,i].plot(PERCENTILES, error_delta, color='red', **marker_style)
            axes[j,i].plot(PERCENTILES, accurate_delta, color='green', **marker_style)
            axes[j,i].legend(['Error', 'Accurate'], loc='upper left')

            # Set y-axis limits and add reference line
            axes[i,j].set_ylim(bottom=-1/3, top=1/2)
            axes[j,i].set_ylim(bottom=-1/3, top=1/2)
            axes[i,j].axhline(y=0, color='gray', linestyle='--', alpha=0.5) 
            axes[j,i].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # Add condition labels
            axes[i,j].text(50, -0.27, 
                          f'{CONDITION_NAMES[conditions[j]]} - {CONDITION_NAMES[conditions[i]]}', 
                          ha='center', va='top', fontsize=12)
            
            axes[j,i].text(50, -0.27, 
                          f'{CONDITION_NAMES[conditions[j]]} - {CONDITION_NAMES[conditions[i]]}', 
                          ha='center', va='top', fontsize=12)
            
            plt.tight_layout()
            
    # Save the figure
    plt.savefig(OUTPUT_DIR / f'delta_plots_{pnum}.png')


# Main execution
if __name__ == "__main__":
    data_file_path = Path(__file__).parent.parent / 'data' / 'data.csv'

    OUTPUT_DIR = Path(__file__).parent.parent / 'output'
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists

    # Check if the data file exists
    if not data_file_path.exists():
        print(f"Error: Data file not found at {data_file_path}")
        print("Please ensure 'data.csv' is in a 'data' folder at the root of your project,")
        print("e.g., if your script is in 'project/code/sdt_ddm.py', data.csv should be in 'project/data/data.csv'.")
        exit() # Exit the script if the data file is not found
    else:
        print(f"Using data file: {data_file_path}")
    
    # Execute SDT Analysis
    print("\n--- Running SDT Analysis ---")
    sdt_data = read_data(data_file_path, prepare_for='sdt', display=True)
    
    if not sdt_data.empty:
        print("\nApplying Hierarchical SDT Model (this may take a moment)...")
        
        sdt_model = apply_hierarchical_sdt_model(sdt_data)

        # Sample from the posterior
        with sdt_model:
            # Adjust the number of draws and chains as needed for convergence
            trace = pm.sample(draws=2000, tune=1000, chains=2, random_seed=42, return_inferencedata=True)
        
        # Display summary of the posterior (d_prime and criterion parameters)
        print("\nSDT Model Summary:")
        print(az.summary(trace, var_names=['mean_d_prime', 'stdev_d_prime', 'mean_criterion', 'stdev_criterion']))
        
        # Plot posterior distributions
        az.plot_trace(trace, var_names=['mean_d_prime', 'mean_criterion'])
        plt.suptitle("Posterior Distributions of Group-Level SDT Parameters")
        
        # Save plots to output directory (already defined above)
        plt.savefig(OUTPUT_DIR / 'sdt_posterior_plots.png')
        plt.close() # Close the plot to prevent it from displaying in the console
        print(f"SDT posterior plots saved to {OUTPUT_DIR / 'sdt_posterior_plots.png'}")

    else:
        print("SDT data is empty. Skipping model application.")

    # Execute Delta Plot Analysis
    print("\n--- Running Delta Plot Analysis ---")
    dp_data = read_data(data_file_path, prepare_for='delta plots', display=True)

    if not dp_data.empty:
        # Draw delta plots for each participant
        for pnum in dp_data['pnum'].unique():
            print(f"\nDrawing delta plots for Participant {pnum}...")
            draw_delta_plots(dp_data, pnum)
            plt.close() # Close the plot after saving
            print(f"Delta plots for Participant {pnum} saved to {OUTPUT_DIR / f'delta_plots_{pnum}.png'}")
    else:
        print("Delta plot data is empty. Skipping delta plot generation.")


# Main execution
if __name__ == "__main__":
    data_path = Path(__file__).parent.parent / 'data' / 'data.csv' 
    sdt_data = read_data(data_path, prepare_for='sdt', display=True)

    sdt_model = apply_hierarchical_sdt_model(sdt_data)

    print("\nSampling from the SDT model posterior...")
    with sdt_model:
        trace = pm.sample(draws=2000, tune=2000, chains=4, cores=4, random_seed=42)
    print("Sampling complete.")

    # Convergence
    print("\n--- Model Convergence Summary ---")
    convergence_summary = az.summary(trace, var_names=[
        'mean_d_prime_intercept', 'effect_stimulus_type_dprime', 'effect_difficulty_dprime',
        'stdev_d_prime', 'mean_criterion_intercept', 'effect_stimulus_type_criterion',
        'effect_difficulty_criterion', 'stdev_criterion'
    ])
    print(convergence_summary)

    # Save the convergence summary to a CSV file for later review
    OUTPUT_DIR = Path(__file__).parent.parent.parent / 'output'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    convergence_summary.to_csv(OUTPUT_DIR / 'sdt_model_convergence_summary.csv')
    print(f"\nConvergence summary saved to: {OUTPUT_DIR / 'sdt_model_convergence_summary.csv'}")

    # Posterior Distributions 
    print("\n--- Displaying Posterior Distributions ---")
    az.plot_posterior(trace, var_names=[
        'mean_d_prime_intercept', 'effect_stimulus_type_dprime', 'effect_difficulty_dprime',
        'mean_criterion_intercept', 'effect_stimulus_type_criterion', 'effect_difficulty_criterion'
    ])
    plt.suptitle("Posterior Distributions of SDT Model Parameters", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'sdt_posterior_distributions.png')
    print(f"Posterior distributions plot saved to: {OUTPUT_DIR / 'sdt_posterior_distributions.png'}")
    plt.show() # Display the plot if running interactively

    # Delta Plots
    delta_plot_data = read_data(data_path, prepare_for='delta plots', display=True)
    print("\n--- Drawing Delta Plots ---")
    participant_to_plot = sdt_data['pnum'].unique()[0] # Plot for the first participant
    draw_delta_plots(delta_plot_data, participant_to_plot)
    print(f"Delta plots for participant {participant_to_plot} saved to: {OUTPUT_DIR / f'delta_plots_{participant_to_plot}.png'}")

    print("\nAnalysis complete. Check 'output' folder!")
