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
                
                # Ensure we have data for both signal and noise trials for a valid SDT calculation
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
                # Only add if there's enough data to calculate percentiles
                if not overall_rt.empty:
                    dp_data = pd.concat([dp_data, pd.DataFrame({
                        'pnum': [pnum],
                        'condition': [condition],
                        'mode': ['overall'],
                        **{f'p{p}': [np.percentile(overall_rt, p)] for p in PERCENTILES}
                    })], ignore_index=True) # Use ignore_index=True with pd.concat
                
                # Calculate percentiles for accurate responses
                accurate_rt = c_data[c_data['accuracy'] == 1]['rt']
                if not accurate_rt.empty:
                    dp_data = pd.concat([dp_data, pd.DataFrame({
                        'pnum': [pnum],
                        'condition': [condition],
                        'mode': ['accurate'],
                        **{f'p{p}': [np.percentile(accurate_rt, p)] for p in PERCENTILES}
                    })], ignore_index=True)
                
                # Calculate percentiles for error responses
                error_rt = c_data[c_data['accuracy'] == 0]['rt']
                if not error_rt.empty:
                    dp_data = pd.concat([dp_data, pd.DataFrame({
                        'pnum': [pnum],
                        'condition': [condition],
                        'mode': ['error'],
                        **{f'p{p}': [np.percentile(error_rt, p)] for p in PERCENTILES}
                    })], ignore_index=True)
                
        if display:
            print("\nDelta plots data:")
            print(dp_data)
            
        data = dp_data # Assign the collected dp_data to 'data'
        if data.empty:
            print("\nWARNING: Empty Delta Plot data generated!")

    return data


def apply_hierarchical_sdt_model(data):
    """Apply a hierarchical Signal Detection Theory model using PyMC.
    
    This function implements a Bayesian hierarchical model for SDT analysis,
    allowing for both group-level and individual-level parameter estimation.
    It includes stimulus type and difficulty as predictors, and their interaction.
    
    Args:
        data: DataFrame containing SDT summary statistics
        
    Returns:
        PyMC model object
    """
    unique_pnums = data['pnum'].unique()
    P = len(unique_pnums)
    unique_conditions = data['condition'].unique()
    C = len(unique_conditions)
    
    stimulus_type_conditions = np.array([c % 2 for c in unique_conditions])      # 0 for Simple, 1 for Complex
    difficulty_conditions = np.array([c // 2 for c in unique_conditions])         # 0 for Easy, 1 for Hard
    
    # Interaction term: product of stimulus_type and difficulty
    interaction_conditions = stimulus_type_conditions * difficulty_conditions # 0 for Simple, 1 for Complex; 0 for Easy, 1 for Hard
    
    # Create mappings from original pnum and condition values to 0-indexed PyMC indices
    pnum_to_idx = {p: i for i, p in enumerate(unique_pnums)}
    condition_to_idx = {c: i for i, c in enumerate(unique_conditions)}

    # Convert participant_id and condition in data to 0-indexed values for direct use
    pnum_data_indexed = data['pnum'].map(pnum_to_idx).values
    condition_data_indexed = data['condition'].map(condition_to_idx).values

    # Define coordinates for cleaner output
    coords = {
        'pnum_idx': unique_pnums, # Use actual participant IDs for better readability
        'condition_idx': [CONDITION_NAMES[c_val] for c_val in unique_conditions] # Use descriptive names
    }

    with pm.Model(coords=coords) as sdt_model:
        # Group-level parameters for effects of stimulus type and difficulty
        
        # d_prime effects
        mean_d_prime_intercept = pm.Normal('mean_d_prime_intercept', mu=0.0, sigma=1.0)
        effect_stimulus_type_dprime = pm.Normal('effect_stimulus_type_dprime', mu=0.0, sigma=0.5)
        effect_difficulty_dprime = pm.Normal('effect_difficulty_dprime', mu=0.0, sigma=0.5)
        effect_interaction_dprime = pm.Normal('effect_interaction_dprime', mu=0.0, sigma=0.5)
        stdev_d_prime_overall = pm.HalfNormal('stdev_d_prime_overall', sigma=1.0)
        
        # criterion effects
        mean_criterion_intercept = pm.Normal('mean_criterion_intercept', mu=0.0, sigma=1.0)
        effect_stimulus_type_criterion = pm.Normal('effect_stimulus_type_criterion', mu=0.0, sigma=0.5)
        effect_difficulty_criterion = pm.Normal('effect_difficulty_criterion', mu=0.0, sigma=0.5)
        effect_interaction_criterion = pm.Normal('effect_interaction_criterion', mu=0.0, sigma=0.5)
        stdev_criterion_overall = pm.HalfNormal('stdev_criterion_overall', sigma=1.0)

        # Define the mean d_prime and criterion for *each condition (C)*
        mean_d_prime = pm.Deterministic(
            'mean_d_prime', 
            mean_d_prime_intercept +
            effect_stimulus_type_dprime * stimulus_type_conditions +
            effect_difficulty_dprime * difficulty_conditions +
            effect_interaction_dprime * interaction_conditions, 
            dims=('condition_idx',)
        )
        
        mean_criterion = pm.Deterministic(
            'mean_criterion', 
            mean_criterion_intercept +
            effect_stimulus_type_criterion * stimulus_type_conditions +
            effect_difficulty_criterion * difficulty_conditions +
            effect_interaction_criterion * interaction_conditions, 
            dims=('condition_idx',)
        )
        
        # Individual-level parameters (P, C)
        d_prime = pm.Normal('d_prime',
                            mu=mean_d_prime,
                            sigma=stdev_d_prime_overall,
                            dims=('pnum_idx', 'condition_idx'))

        criterion = pm.Normal('criterion',
                             mu=mean_criterion,
                             sigma=stdev_criterion_overall,
                             dims=('pnum_idx', 'condition_idx'))
        
        # Likelihood for signal trials
        hit_rate = pm.math.invlogit(d_prime[pnum_data_indexed, condition_data_indexed] - criterion[pnum_data_indexed, condition_data_indexed])
        false_alarm_rate = pm.math.invlogit(-criterion[pnum_data_indexed, condition_data_indexed])
                
        # Likelihood for signal trials
        pm.Binomial('hit_obs', 
                   n=data['nSignal'].values, 
                   p=hit_rate, 
                   observed=data['hits'].values)
        
        # Likelihood for noise trials
        pm.Binomial('false_alarm_obs', 
                   n=data['nNoise'].values, 
                   p=false_alarm_rate, 
                   observed=data['false_alarms'].values)
    
    return sdt_model


def draw_delta_plots(data, pnum):
    """Draw delta plots comparing RT distributions between condition pairs.
    
    Creates a matrix of delta plots where:
    - Upper triangle shows overall RT distribution differences (black line)
    - Lower triangle shows RT differences split by correct/error responses (red/green lines)
    
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
    
    # Create output directory relative to where script is run
    OUTPUT_DIR = Path(__file__).parent.parent / 'output'
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
                
            # Diagonal: Turn off axes and add condition name
            if i == j:
                axes[i,j].axis('off')
                axes[i,j].text(0.5, 0.5, CONDITION_NAMES[conditions[i]], 
                               ha='center', va='center', fontsize=14, fontweight='bold')
                continue # Skip to next iteration

            # Upper Triangle (i < j): Overall RT distribution differences (black line)
            # This is where OVERALL RT differences typically go.
            if i < j:
                cmask1 = data['condition'] == cond1
                cmask2 = data['condition'] == cond2
                overall_mask = data['mode'] == 'overall'
                
                # Check if data exists for the conditions and mode before attempting percentile calculation
                if not data[cmask1 & overall_mask].empty and not data[cmask2 & overall_mask].empty:
                    quantiles1 = [data[cmask1 & overall_mask][f'p{p}'].iloc[0] for p in PERCENTILES]
                    quantiles2 = [data[cmask2 & overall_mask][f'p{p}'].iloc[0] for p in PERCENTILES]
                    overall_delta = np.array(quantiles2) - np.array(quantiles1)
                    axes[i,j].plot(PERCENTILES, overall_delta, color='black', **marker_style)
                else:
                    axes[i,j].text(50, 0, "No data", ha='center', va='center', fontsize=10, color='gray')

                axes[i,j].set_ylim(bottom=-1/3, top=1/2) # Set common y-axis limits
                axes[i,j].axhline(y=0, color='gray', linestyle='--', alpha=0.5) 
                axes[i,j].set_title(f'{CONDITION_NAMES[conditions[j]]} - {CONDITION_NAMES[conditions[i]]}', 
                                    fontsize=12, pad=10)

            # Lower Triangle (i > j): RT differences split by correct/error responses (red/green lines)
            # This is where accuracy-dependent plots typically go.
            elif i > j:
                cmask1 = data['condition'] == cond1
                cmask2 = data['condition'] == cond2
                error_mask = data['mode'] == 'error'
                accurate_mask = data['mode'] == 'accurate'

                plotted_data = False
                if not data[cmask1 & error_mask].empty and not data[cmask2 & error_mask].empty:
                    error_quantiles1 = [data[cmask1 & error_mask][f'p{p}'].iloc[0] for p in PERCENTILES]
                    error_quantiles2 = [data[cmask2 & error_mask][f'p{p}'].iloc[0] for p in PERCENTILES]
                    error_delta = np.array(error_quantiles2) - np.array(error_quantiles1)
                    axes[i,j].plot(PERCENTILES, error_delta, color='red', **marker_style)
                    plotted_data = True
                
                if not data[cmask1 & accurate_mask].empty and not data[cmask2 & accurate_mask].empty:
                    accurate_quantiles1 = [data[cmask1 & accurate_mask][f'p{p}'].iloc[0] for p in PERCENTILES]
                    accurate_quantiles2 = [data[cmask2 & accurate_mask][f'p{p}'].iloc[0] for p in PERCENTILES]
                    accurate_delta = np.array(accurate_quantiles2) - np.array(accurate_quantiles1)
                    axes[i,j].plot(PERCENTILES, accurate_delta, color='green', **marker_style)
                    plotted_data = True

                if plotted_data:
                    axes[i,j].legend(['Error', 'Accurate'], loc='upper left')
                else:
                    axes[i,j].text(50, 0, "No data", ha='center', va='center', fontsize=10, color='gray')

                axes[i,j].set_ylim(bottom=-1/3, top=1/2) # Set common y-axis limits
                axes[i,j].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                axes[i,j].set_title(f'{CONDITION_NAMES[conditions[j]]} - {CONDITION_NAMES[conditions[i]]}', 
                                    fontsize=12, pad=10)
            
    plt.tight_layout()
            
    # Save the figure
    plt.savefig(OUTPUT_DIR / f'delta_plots_{pnum}.png')


def analyze_results(trace, data, OUTPUT_DIR):
    """
    Analyze and visualize the posterior samples from the SDT model.
    """
    # --- Check Convergence & Main Effect Summary ---
    print("\n" + "-"*30 + "\n--- SDT Model Convergence & Main Effect Summary ---\n" + "-"*30)
    convergence_summary = az.summary(trace, var_names=[
        'mean_d_prime_intercept', 'effect_stimulus_type_dprime', 'effect_difficulty_dprime',
        'effect_interaction_dprime',
        'stdev_d_prime_overall',
        'mean_criterion_intercept', 'effect_stimulus_type_criterion',
        'effect_difficulty_criterion', 'effect_interaction_criterion', 
        'stdev_criterion_overall'
    ])
    print(convergence_summary)

    # Save the convergence summary to a CSV file
    convergence_summary.to_csv(OUTPUT_DIR / 'convergence_summary.csv')
    print(f"\nConvergence summary saved to: {OUTPUT_DIR / 'convergence_summary.csv'}")

    # --- Display Posterior Distributions (Trace Plots) ---
    print("\n" + "\n--- Posterior Trace Plots ---\n")
    az.plot_trace(trace, var_names=[
        'mean_d_prime_intercept', 'effect_stimulus_type_dprime', 'effect_difficulty_dprime',
        'effect_interaction_dprime', 
        'mean_criterion_intercept', 'effect_stimulus_type_criterion',
        'effect_difficulty_criterion', 'effect_interaction_criterion' 
    ])
    plt.suptitle("Posterior Trace Plots of SDT Model Parameters", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'trace_plots.png')
    print(f"SDT trace plots saved to: {OUTPUT_DIR / 'trace_plots.png'}")
    plt.close()

    # --- Display Posterior Distributions (Density Plots) ---
    print("\n" + "\n--- Posterior Density Plots ---\n")
    az.plot_posterior(trace, var_names=[
        'mean_d_prime_intercept', 'effect_stimulus_type_dprime', 'effect_difficulty_dprime',
        'effect_interaction_dprime', 
        'mean_criterion_intercept', 'effect_stimulus_type_criterion',
        'effect_difficulty_criterion', 'effect_interaction_criterion' 
    ])
    plt.suptitle("Posterior Density Plots of SDT Model Parameters", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'posterior_density_plots.png')
    print(f"Posterior density plots saved to: {OUTPUT_DIR / 'posterior_density_plots.png'}")
    plt.close()

    # --- Analyze Condition-Specific SDT Parameters ---
    print("\n" + "-"*30 + "\n--- Condition-Specific SDT Parameters ---\n" + "-"*30)
    # This will show the estimated d' and criterion for each of your 4 conditions
    condition_sdt_summary = az.summary(trace, var_names=['mean_d_prime', 'mean_criterion'])
    print(condition_sdt_summary)
    condition_sdt_summary.to_csv(OUTPUT_DIR / 'condition_parameters_summary.csv')
    print(f"\nCondition-specific SDT parameters saved to: {OUTPUT_DIR / 'condition_parameters_summary.csv'}")

    # Plot condition-specific d' and criterion
    az.plot_posterior(trace, var_names=['mean_d_prime', 'mean_criterion'],
                      hdi_prob=0.94, figsize=(10, 6))
    plt.suptitle("Posterior Distributions of Mean d' and Criterion per Condition", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'condition_posterior_plots.png')
    print(f"Condition-specific posterior plots saved to: {OUTPUT_DIR / 'condition_posterior_plots.png'}")
    plt.close()

    print("\n" + "-"*30 + "\n--- Person-Specific SDT Parameter Estimates ---\n" + "-"*30)
    individual_sdt_summary = az.summary(trace, var_names=['d_prime', 'criterion'])
    print(individual_sdt_summary)
    individual_sdt_summary.to_csv(OUTPUT_DIR / 'individual_parameters_summary.csv')
    print(f"\nPerson-specific SDT parameters saved to: {OUTPUT_DIR / 'individual_parameters_summary.csv'}")

    
    # --- Derived Comparisons (e.g., Simple vs Complex within Easy/Hard) ---
    print("\n" + "-"*30 + "\n--- Derived SDT Parameter Comparisons ---\n" + "-"*30)
    
    # Access the relevant posterior samples for calculations
    mean_d_prime_posterior_samples = trace.posterior['mean_d_prime']
    mean_criterion_posterior_samples = trace.posterior['mean_criterion']

    # Calculate derived parameters and add them directly to the posterior group of the InferenceData
    trace.posterior['d_prime_effect_stim_type_easy'] = (
        mean_d_prime_posterior_samples.sel(condition_idx='Easy Complex') - 
        mean_d_prime_posterior_samples.sel(condition_idx='Easy Simple')
    )
    trace.posterior['criterion_effect_stim_type_easy'] = (
        mean_criterion_posterior_samples.sel(condition_idx='Easy Complex') - 
        mean_criterion_posterior_samples.sel(condition_idx='Easy Simple')
    )

    # Effect of Stimulus Type for HARD trials: (Hard Complex) - (Hard Simple)
    trace.posterior['d_prime_effect_stim_type_hard'] = (
        mean_d_prime_posterior_samples.sel(condition_idx='Hard Complex') - 
        mean_d_prime_posterior_samples.sel(condition_idx='Hard Simple')
    )
    trace.posterior['criterion_effect_stim_type_hard'] = (
        mean_criterion_posterior_samples.sel(condition_idx='Hard Complex') - 
        mean_criterion_posterior_samples.sel(condition_idx='Hard Simple')
    )

    # Effect of Trial Difficulty (Hard - Easy) for each Stimulus Type:
    trace.posterior['d_prime_effect_difficulty_simple'] = (
        mean_d_prime_posterior_samples.sel(condition_idx='Hard Simple') - 
        mean_d_prime_posterior_samples.sel(condition_idx='Easy Simple')
    )
    trace.posterior['criterion_effect_difficulty_simple'] = (
        mean_criterion_posterior_samples.sel(condition_idx='Hard Simple') - 
        mean_criterion_posterior_samples.sel(condition_idx='Easy Simple')
    )
    
    # Effect of Difficulty for COMPLEX stimuli: (Hard Complex) - (Easy Complex)
    trace.posterior['d_prime_effect_difficulty_complex'] = (
        mean_d_prime_posterior_samples.sel(condition_idx='Hard Complex') - 
        mean_d_prime_posterior_samples.sel(condition_idx='Easy Complex')
    )
    trace.posterior['criterion_effect_difficulty_complex'] = (
        mean_criterion_posterior_samples.sel(condition_idx='Hard Complex') - 
        mean_criterion_posterior_samples.sel(condition_idx='Easy Complex')
    )

    # Define the list of derived variable names for summary and plotting
    derived_vars_dprime = [
        'd_prime_effect_stim_type_easy', 'd_prime_effect_stim_type_hard',
        'd_prime_effect_difficulty_simple', 'd_prime_effect_difficulty_complex'
    ]
    derived_vars_criterion = [
        'criterion_effect_stim_type_easy', 'criterion_effect_stim_type_hard',
        'criterion_effect_difficulty_simple', 'criterion_effect_difficulty_complex'
    ]

    # Display summaries of these derived parameters
    print("\nSummary of Derived d' Effects (Conditional):")
    derived_dprime_summary = az.summary(trace, var_names=derived_vars_dprime, hdi_prob=0.94)
    print(derived_dprime_summary)
    derived_dprime_summary.to_csv(OUTPUT_DIR / 'derived_dprime_effects.csv')

    print("\nSummary of Derived Criterion Effects (Conditional):")
    derived_criterion_summary = az.summary(trace, var_names=derived_vars_criterion, hdi_prob=0.94)
    print(derived_criterion_summary)
    derived_criterion_summary.to_csv(OUTPUT_DIR / 'derived_criterion_effects.csv')


# Main execution block
if __name__ == "__main__":
    data_file_path = Path(__file__).parent.parent / 'data' / 'data.csv'

    OUTPUT_DIR = Path(__file__).parent.parent / 'output'
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists

    if not data_file_path.exists():
        print(f"Error: Data file not found at {data_file_path}")
        print("Please ensure 'data.csv' is in a 'data' folder at the root of your project,")
        print("e.g., if your script is in 'project/code/sdt_ddm.py', data.csv should be in 'project/data/data.csv'.")
        exit()
    else:
        print(f"Using data file: {data_file_path}")
    
    # SDT Analysis
    print("\n" + "="*30 + "\n--- Running SDT Analysis ---\n" + "="*30)
    sdt_data = read_data(data_file_path, prepare_for='sdt', display=True)
    
    if not sdt_data.empty:
        print("\nApplying Hierarchical SDT Model (this may take a moment)...")
        
        sdt_model = apply_hierarchical_sdt_model(sdt_data)

        # Sample from the posterior distribution
        print("\nSampling from the SDT model posterior...")
        with sdt_model:
            trace = pm.sample(draws=2000, tune=2000, chains=4, cores=4, random_seed=42, return_inferencedata=True)
        print("Sampling complete.")

        # Pass the trace, the original sdt_data (which contains the condition mapping), and OUTPUT_DIR
        analyze_results(trace, sdt_data, OUTPUT_DIR) 
        
    else:
        print("SDT data is empty. Skipping model application and analysis.")

    # Delta Plot Analysis 
    print("\n" + "="*30 + "\n--- Running Delta Plot Analysis ---\n" + "="*30)
    dp_data = read_data(data_file_path, prepare_for='delta plots', display=True)

    if not dp_data.empty:
        if not dp_data['pnum'].empty:
            for pnum in dp_data['pnum'].unique():
                print(f"\nDrawing delta plots for Participant {pnum}...")
                draw_delta_plots(dp_data, pnum)
                plt.close()
                print(f"Delta plots for Participant {pnum} saved to {OUTPUT_DIR / f'delta_plots_{pnum}.png'}")
        else:
            print("No participants found in delta plot data to draw plots for.")
    else:
        print("Delta plot data is empty. Skipping delta plot generation.")

    print("\n" + "="*30 + "\n--- All Analyses Complete ---\n" + "="*30)
    print(f"Check the '{OUTPUT_DIR.name}' folder for generated files!")

