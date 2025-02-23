import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress tracking

# Configuration
RESULTS_DIR = "Simulation_results"
PLOT_DIR = "Plots"
os.makedirs(PLOT_DIR, exist_ok=True)


def load_simulation_results(results_dir):
    """Load all JSON results from simulation directory"""
    results = []
    for filename in tqdm(os.listdir(results_dir), desc="Loading JSON files"):
        if filename.startswith("run") and filename.endswith(".json"):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Convert lists to numpy arrays for efficient computation
                data['active_status_over_time'] = np.array(data['active_status_over_time'], dtype=bool)
                data['quality_over_time'] = np.array(data['quality_over_time'], dtype=float)
                data['is_CC_protected'] = np.array(data['is_CC_protected'], dtype=bool)
                results.append(data)
    return results


def calculate_retention_metrics(results):
    """Calculate retention metrics across all simulations"""
    max_timesteps = max(len(r['active_status_over_time']) for r in results)
    n_simulations = len(results)

    # Initialize accumulation arrays
    protected_retention = np.zeros((n_simulations, max_timesteps))
    unprotected_retention = np.zeros((n_simulations, max_timesteps))

    for sim_idx, result in enumerate(results):
        active_status = result['active_status_over_time']
        is_protected = result['is_CC_protected']
        n_protected = np.sum(is_protected)
        n_unprotected = len(is_protected) - n_protected

        for t in range(len(active_status)):
            # Calculate fraction active for this timestep
            protected_active = np.sum(active_status[t] & is_protected)
            unprotected_active = np.sum(active_status[t] & ~is_protected)

            protected_retention[sim_idx, t] = protected_active / n_protected if n_protected > 0 else 0
            unprotected_retention[sim_idx, t] = unprotected_active / n_unprotected if n_unprotected > 0 else 0

    return protected_retention, unprotected_retention


def calculate_quality_metrics(results):
    """Calculate quality metrics across all simulations"""
    max_timesteps = max(len(r['quality_over_time']) for r in results)
    n_simulations = len(results)

    # Initialize accumulation arrays
    protected_active_quality = np.zeros((n_simulations, max_timesteps))
    unprotected_active_quality = np.zeros((n_simulations, max_timesteps))
    protected_dropout_quality = []
    unprotected_dropout_quality = []

    for sim_idx, result in enumerate(results):
        quality = result['quality_over_time']
        active_status = result['active_status_over_time']
        is_protected = result['is_CC_protected']

        # Track last quality before dropout for each CC
        dropout_qualities = {}
        for cc_idx in range(quality.shape[1]):
            active_timesteps = np.where(active_status[:, cc_idx])[0]
            if len(active_timesteps) < len(active_status):  # Fixed syntax
                dropout_time = len(active_timesteps)
                dropout_qualities[cc_idx] = quality[dropout_time - 1, cc_idx]

        # Calculate averages per timestep
        for t in range(len(quality)):
            protected_active = is_protected & active_status[t]
            unprotected_active = ~is_protected & active_status[t]

            if np.any(protected_active):
                protected_active_quality[sim_idx, t] = np.mean(quality[t, protected_active])
            if np.any(unprotected_active):
                unprotected_active_quality[sim_idx, t] = np.mean(quality[t, unprotected_active])

        # Collect dropout qualities
        for cc_idx, q in dropout_qualities.items():
            if is_protected[cc_idx]:
                protected_dropout_quality.append(q)
            else:
                unprotected_dropout_quality.append(q)

    return (
        protected_active_quality,
        unprotected_active_quality,
        np.array(protected_dropout_quality),
        np.array(unprotected_dropout_quality)
    )

def plot_retention(protected_retention, unprotected_retention):
    """Plot creator retention over time with confidence intervals"""
    plt.figure(figsize=(10, 6))

    # Calculate statistics across simulations
    timesteps = protected_retention.shape[1]
    x = np.arange(timesteps)

    for data, label in zip([protected_retention, unprotected_retention],
                           ['Protected', 'Unprotected']):
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)

        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean - 1.96 * std, mean + 1.96 * std, alpha=0.2)

    plt.xlabel("Iteration")
    plt.ylabel("Fraction of Active Creators")
    plt.title("Creator Retention Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "retention_plot.png"))
    plt.show()
    plt.close()


def plot_quality_evolution(active_protected, active_unprotected,
                           dropout_protected, dropout_unprotected):
    """Plot quality evolution with active/dropout breakdown"""
    plt.figure(figsize=(10, 6))

    # Active quality plots
    for data, label in zip([active_protected, active_unprotected],
                           ['Protected (Active)', 'Unprotected (Active)']):
        mean = np.nanmean(data, axis=0)
        plt.plot(mean, label=label, linestyle='-')

    # Dropout quality markers
    if len(dropout_protected) > 0:
        plt.hlines(np.mean(dropout_protected), 0, active_protected.shape[1],
                   colors='C0', linestyles='--', label='Protected (Dropout)')
    if len(dropout_unprotected) > 0:
        plt.hlines(np.mean(dropout_unprotected), 0, active_unprotected.shape[1],
                   colors='C1', linestyles='--', label='Unprotected (Dropout)')

    plt.xlabel("Iteration")
    plt.ylabel("Average Quality")
    plt.title("Quality Evolution Over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOT_DIR, "quality_evolution_plot.png"))
    plt.show()
    plt.close()


# Add these new functions to the analysis.py script

### PLOT 3: Activity Score Distribution ----------------------------------------
def calculate_activity_scores(results, alpha=1.0, beta=2.0):
    """Calculate activity scores across all simulations"""
    activity_data = {
        'protected': {'early': [], 'mid': [], 'pre_dropout': []},
        'unprotected': {'early': [], 'mid': [], 'pre_dropout': []}
    }

    for result in tqdm(results, desc="Processing activity scores"):
        # Extract required metrics from corrected keys
        posts = np.array(result['posts_recent'])
        followers_current = np.array(result['followers_current'])
        total_followers = np.array(result['total_followers'])
        active_status = result['active_status_over_time']
        is_protected = result['is_CC_protected']

        # Calculate activity scores
        with np.errstate(divide='ignore', invalid='ignore'):
            engagement = followers_current / np.maximum(1, total_followers)
            activity_scores = alpha * posts + beta * engagement

        # Transpose to get (timesteps, CCs) shape
        activity_scores = activity_scores.T

        # Sample at key phases
        n_timesteps = activity_scores.shape[0]
        for cc_idx in range(activity_scores.shape[1]):
            group = 'protected' if is_protected[cc_idx] else 'unprotected'

            # Early phase (first 20% of simulation)
            early_t = min(5, n_timesteps - 1)
            activity_data[group]['early'].append(activity_scores[early_t, cc_idx])

            # Mid phase (50% point)
            mid_t = min(n_timesteps // 2, n_timesteps - 1)
            activity_data[group]['mid'].append(activity_scores[mid_t, cc_idx])

            # Pre-dropout phase (last active timestep)
            active_ts = np.where(active_status[:, cc_idx])[0]
            if len(active_ts) > 0 and len(active_ts) < n_timesteps:
                pre_dropout_t = active_ts[-1]
                activity_data[group]['pre_dropout'].append(
                    activity_scores[pre_dropout_t, cc_idx]
                )

    return activity_data


def plot_activity_distribution(activity_data):
    """Plot activity score distributions at different phases"""
    phases = ['early', 'mid', 'pre_dropout']
    titles = ['Early Phase (t=5)', 'Mid Phase (t=50)', 'Pre-Dropout Phase']

    plt.figure(figsize=(15, 5))
    for idx, (phase, title) in enumerate(zip(phases, titles), 1):
        plt.subplot(1, 3, idx)

        # Extract data
        prot = activity_data['protected'][phase]
        unprot = activity_data['unprotected'][phase]

        # Plot violin plots
        parts = plt.violinplot([prot, unprot], showmeans=True, showmedians=True)

        # Style customization
        for pc in parts['bodies']:
            pc.set_facecolor('C0')
            pc.set_edgecolor('black')
            pc.set_alpha(0.7)

        parts['cbars'].set_color('black')
        parts['cmins'].set_color('black')
        parts['cmaxes'].set_color('black')
        parts['cmeans'].set_color('red')
        parts['cmedians'].set_color('blue')

        plt.xticks([1, 2], ['Protected', 'Unprotected'])
        plt.title(title)
        plt.ylabel('Activity Score')
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "activity_distribution_plot.png"))
    plt.show()
    plt.close()


### PLOT 4: Time-to-Dropout Analysis -------------------------------------------
def calculate_dropout_times(results):
    """Calculate dropout times for survival analysis"""
    dropout_data = {
        'protected': {'time': [], 'event': []},
        'unprotected': {'time': [], 'event': []}
    }

    for result in tqdm(results, desc="Processing dropout times"):
        active_status = result['active_status_over_time']
        is_protected = result['is_CC_protected']

        for cc_idx in range(active_status.shape[1]):
            active_ts = np.where(active_status[:, cc_idx])[0]
            event = 1 if len(active_ts) < active_status.shape[0] else 0
            time = len(active_ts)  # Last active timestep

            group = 'protected' if is_protected[cc_idx] else 'unprotected'
            dropout_data[group]['time'].append(time)
            dropout_data[group]['event'].append(event)

    return dropout_data


def plot_survival_curves(dropout_data):
    """Plot survival curves using Kaplan-Meier estimator"""
    from lifelines import KaplanMeierFitter

    plt.figure(figsize=(10, 6))

    kmf = KaplanMeierFitter()

    for group, color in zip(['protected', 'unprotected'], ['C0', 'C1']):
        T = np.array(dropout_data[group]['time'])
        E = np.array(dropout_data[group]['event'])

        kmf.fit(T, event_observed=E, label=group.capitalize())
        kmf.plot_survival_function(ci_show=True, color=color)

    plt.xlabel('Iterations')
    plt.ylabel('Survival Probability')
    plt.title('Survival Analysis of Content Creators')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, "survival_analysis_plot.png"))
    plt.show()
    plt.close()


# Main execution
if __name__ == "__main__":
    # Load and process data
    results = load_simulation_results(RESULTS_DIR)

    # Generate retention plot
    prot_ret, unprot_ret = calculate_retention_metrics(results)
    plot_retention(prot_ret, unprot_ret)

    # Generate quality evolution plot
    (prot_active_q, unprot_active_q,
     prot_dropout_q, unprot_dropout_q) = calculate_quality_metrics(results)
    plot_quality_evolution(prot_active_q, unprot_active_q,
                           prot_dropout_q, unprot_dropout_q)

    # Plot 3: Activity Score Distribution
    activity_data = calculate_activity_scores(results)
    plot_activity_distribution(activity_data)

    # Plot 4: Time-to-Dropout Analysis
    dropout_data = calculate_dropout_times(results)
    plot_survival_curves(dropout_data)