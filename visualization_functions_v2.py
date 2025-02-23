import glob
import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any
import json


class SimulationVisualizer:
    def __init__(self, results_folder: str):
        """
        Initialize visualizer with simulation results

        Args:
            results_file: Path to the JSON results file from simulation
        """
        self.results_folder = results_folder
        self.all_results = []
        # Load all JSON files in the folder
        json_files = glob.glob(os.path.join(results_folder, 'run*.json'))
        for file_path in json_files:
            with open(file_path, 'r') as f:
                self.all_results.append(json.load(f))

        # with open(results_file, 'r') as f:
        #     self.data = json.load(f)
        #
        # # Extract key metrics
        # self.active_status = np.array(self.data['active_status_over_time'])
        # self.quality_over_time = np.array(self.data['quality_over_time'])
        # self.is_protected = np.array(self.data['is_CC_protected'])

        # Set style
        plt.style.use('seaborn')

    # def plot_retention(self, save_path: str = None):
    #     """
    #     Plot creator retention over time for protected and unprotected groups
    #     """
    #     num_timesteps = len(self.active_status)
    #
    #     # Calculate fraction active for each group
    #     protected_active = []
    #     unprotected_active = []
    #
    #     for t in range(num_timesteps):
    #         # For protected group
    #         protected_mask = self.is_protected
    #         protected_active.append(
    #             np.mean(self.active_status[t][protected_mask])
    #         )
    #
    #         # For unprotected group
    #         unprotected_mask = ~self.is_protected
    #         unprotected_active.append(
    #             np.mean(self.active_status[t][unprotected_mask])
    #         )
    #
    #     # Create plot
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(range(num_timesteps), protected_active,
    #              label='Protected', color='blue', alpha=0.7)
    #     plt.plot(range(num_timesteps), unprotected_active,
    #              label='Unprotected', color='green', alpha=0.7)
    #     plt.fill_between(range(num_timesteps), protected_active,
    #                      alpha=0.3, color='blue')
    #     plt.fill_between(range(num_timesteps), unprotected_active,
    #                      alpha=0.3, color='green')
    #
    #     plt.xlabel('Time Step')
    #     plt.ylabel('Fraction Active')
    #     plt.title('Creator Retention Over Time')
    #     plt.legend()
    #     plt.grid(True, alpha=0.3)
    #
    #     if save_path:
    #         plt.savefig(save_path)
    #     plt.show()

    # def plot_retention(self, save_path: str = None):
    #     """
    #     Plot creator retention over time split into 4 groups across multiple simulations
    #     """
    #     # Initialize storage for retention rates across all simulations
    #     all_retention_rates = {
    #         'Protected (Top 25%)': [],
    #         'Protected (Bottom 75%)': [],
    #         'Unprotected (Top 25%)': [],
    #         'Unprotected (Bottom 75%)': []
    #     }
    #
    #     # Process each simulation
    #     for sim_data in self.all_results:
    #         # Convert data to numpy arrays
    #         active_status = np.array(sim_data['active_status_over_time'])
    #         quality_over_time = np.array(sim_data['quality_over_time'])
    #         is_protected = np.array(sim_data['is_CC_protected'])
    #
    #         num_timesteps = len(active_status)
    #         initial_qualities = quality_over_time[0]
    #
    #         # Create basic masks
    #         protected_mask = is_protected
    #         unprotected_mask = ~is_protected
    #
    #         # Calculate thresholds for this simulation
    #         protected_qualities = initial_qualities[protected_mask]
    #         unprotected_qualities = initial_qualities[unprotected_mask]
    #
    #         protected_threshold = np.percentile(protected_qualities, 75)
    #         unprotected_threshold = np.percentile(unprotected_qualities, 75)
    #
    #         # Create subgroup masks
    #         protected_top_mask = protected_mask & (initial_qualities >= protected_threshold)
    #         protected_bottom_mask = protected_mask & (initial_qualities < protected_threshold)
    #         unprotected_top_mask = unprotected_mask & (initial_qualities >= unprotected_threshold)
    #         unprotected_bottom_mask = unprotected_mask & (initial_qualities < unprotected_threshold)
    #
    #         # Calculate retention rates for this simulation
    #         sim_retention_rates = {
    #             'Protected (Top 25%)': [],
    #             'Protected (Bottom 75%)': [],
    #             'Unprotected (Top 25%)': [],
    #             'Unprotected (Bottom 75%)': []
    #         }
    #
    #         for t in range(num_timesteps):
    #             current_active = active_status[t]
    #
    #             sim_retention_rates['Protected (Top 25%)'].append(
    #                 np.mean(current_active[protected_top_mask])
    #             )
    #             sim_retention_rates['Protected (Bottom 75%)'].append(
    #                 np.mean(current_active[protected_bottom_mask])
    #             )
    #             sim_retention_rates['Unprotected (Top 25%)'].append(
    #                 np.mean(current_active[unprotected_top_mask])
    #             )
    #             sim_retention_rates['Unprotected (Bottom 75%)'].append(
    #                 np.mean(current_active[unprotected_bottom_mask])
    #             )
    #
    #         # Store rates from this simulation
    #         for group in all_retention_rates:
    #             all_retention_rates[group].append(sim_retention_rates[group])
    #
    #     # Convert lists to numpy arrays for easier calculation
    #     for group in all_retention_rates:
    #         all_retention_rates[group] = np.array(all_retention_rates[group])
    #
    #     # Calculate mean and confidence intervals
    #     retention_stats = {}
    #     for group in all_retention_rates:
    #         rates = all_retention_rates[group]
    #         retention_stats[group] = {
    #             'mean': np.mean(rates, axis=0),
    #             'std': np.std(rates, axis=0),
    #             'ci_lower': np.percentile(rates, 25, axis=0),
    #             'ci_upper': np.percentile(rates, 75, axis=0)
    #         }
    #
    #     # Create plot
    #     plt.figure(figsize=(12, 6))
    #
    #     colors = {
    #         'Protected (Top 25%)': 'darkblue',
    #         'Protected (Bottom 75%)': 'lightblue',
    #         'Unprotected (Top 25%)': 'darkgreen',
    #         'Unprotected (Bottom 75%)': 'lightgreen'
    #     }
    #
    #     styles = {
    #         'Protected (Top 25%)': '-',
    #         'Protected (Bottom 75%)': '--',
    #         'Unprotected (Top 25%)': '-',
    #         'Unprotected (Bottom 75%)': '--'
    #     }
    #
    #     # Plot each group with confidence intervals
    #     for group_name in retention_stats:
    #         stats = retention_stats[group_name]
    #         timesteps = range(len(stats['mean']))
    #
    #         # Plot mean line
    #         plt.plot(
    #             timesteps,
    #             stats['mean'],
    #             label=group_name,
    #             color=colors[group_name],
    #             linestyle=styles[group_name],
    #             linewidth=2
    #         )
    #
    #         # Add confidence interval
    #         plt.fill_between(
    #             timesteps,
    #             stats['ci_lower'],
    #             stats['ci_upper'],
    #             color=colors[group_name],
    #             alpha=0.2
    #         )
    #
    #     plt.xlabel('Time Step')
    #     plt.ylabel('Fraction Active')
    #     plt.title(f'Creator Retention by Initial Quality\n(Averaged over {len(self.all_results)} simulations)')
    #     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #     plt.grid(True, alpha=0.3)
    #
    #     # Add summary statistics
    #     stats_text = []
    #     for group_name, stats in retention_stats.items():
    #         final_mean = stats['mean'][-1]
    #         final_std = stats['std'][-1]
    #         stats_text.append(f"{group_name}:")
    #         stats_text.append(f"  Final: {final_mean:.1%} ± {final_std:.1%}")
    #
    #     plt.figtext(1.15, 0.5, '\n'.join(stats_text),
    #                 bbox=dict(facecolor='white', alpha=0.8))
    #
    #     plt.tight_layout()
    #
    #     if save_path:
    #         plt.savefig(save_path, bbox_inches='tight', dpi=300)
    #     plt.show()
    #
    #     # Print detailed statistics
    #     print("\nDetailed Retention Statistics:")
    #     print("-" * 40)
    #     for group_name, stats in retention_stats.items():
    #         print(f"\n{group_name}:")
    #         print(f"Final retention rate: {stats['mean'][-1]:.1%} ± {stats['std'][-1]:.1%}")
    #         print(f"Average retention rate: {np.mean(stats['mean']):.1%}")
    #
    #         # Time to 50% dropout (across all simulations)
    #         half_life_times = []
    #         for sim_rates in all_retention_rates[group_name]:
    #             half_life = next((i for i, r in enumerate(sim_rates) if r <= 0.5), None)
    #             if half_life is not None:
    #                 half_life_times.append(half_life)
    #
    #         if half_life_times:
    #             mean_half_life = np.mean(half_life_times)
    #             std_half_life = np.std(half_life_times)
    #             print(f"Time to 50% retention: {mean_half_life:.1f} ± {std_half_life:.1f} timesteps")

    # def plot_retention(self, save_path: str = None):
    #     """
    #     Plot creator retention over time split into 4 groups across multiple simulations
    #     """
    #     # First, find the minimum number of timesteps across all simulations
    #     min_timesteps = float('inf')
    #     for sim_data in self.all_results:
    #         active_status = np.array(sim_data['active_status_over_time'])
    #         min_timesteps = min(min_timesteps, len(active_status))
    #
    #     # Initialize storage for retention rates across all simulations
    #     all_retention_rates = {
    #         'Protected (Top 25%)': [],
    #         'Protected (Bottom 75%)': [],
    #         'Unprotected (Top 25%)': [],
    #         'Unprotected (Bottom 75%)': []
    #     }
    #
    #     # Process each simulation
    #     for sim_data in self.all_results:
    #         # Convert data to numpy arrays
    #         active_status = np.array(sim_data['active_status_over_time'])[:min_timesteps]  # Truncate to min_timesteps
    #         quality_over_time = np.array(sim_data['quality_over_time'])
    #         is_protected = np.array(sim_data['is_CC_protected'])
    #
    #         initial_qualities = quality_over_time[0]
    #
    #         # Create basic masks
    #         protected_mask = is_protected
    #         unprotected_mask = ~is_protected
    #
    #         # Calculate thresholds for this simulation
    #         protected_qualities = initial_qualities[protected_mask]
    #         unprotected_qualities = initial_qualities[unprotected_mask]
    #
    #         if len(protected_qualities) > 0:
    #             protected_threshold = np.percentile(protected_qualities, 75)
    #         else:
    #             protected_threshold = float('inf')
    #
    #         if len(unprotected_qualities) > 0:
    #             unprotected_threshold = np.percentile(unprotected_qualities, 75)
    #         else:
    #             unprotected_threshold = float('inf')
    #
    #         # Create subgroup masks
    #         protected_top_mask = protected_mask & (initial_qualities >= protected_threshold)
    #         protected_bottom_mask = protected_mask & (initial_qualities < protected_threshold)
    #         unprotected_top_mask = unprotected_mask & (initial_qualities >= unprotected_threshold)
    #         unprotected_bottom_mask = unprotected_mask & (initial_qualities < unprotected_threshold)
    #
    #         # Calculate retention rates for this simulation
    #         sim_retention_rates = {
    #             'Protected (Top 25%)': [],
    #             'Protected (Bottom 75%)': [],
    #             'Unprotected (Top 25%)': [],
    #             'Unprotected (Bottom 75%)': []
    #         }
    #
    #         for t in range(min_timesteps):
    #             current_active = active_status[t]
    #
    #             # Calculate means, handling empty groups
    #             for group_name, mask in [
    #                 ('Protected (Top 25%)', protected_top_mask),
    #                 ('Protected (Bottom 75%)', protected_bottom_mask),
    #                 ('Unprotected (Top 25%)', unprotected_top_mask),
    #                 ('Unprotected (Bottom 75%)', unprotected_bottom_mask)
    #             ]:
    #                 if np.any(mask):  # If group is not empty
    #                     rate = np.mean(current_active[mask])
    #                 else:
    #                     rate = 0.0  # or np.nan if you prefer
    #                 sim_retention_rates[group_name].append(rate)
    #
    #         # Store rates from this simulation
    #         for group in all_retention_rates:
    #             all_retention_rates[group].append(sim_retention_rates[group])
    #
    #     # Convert lists to numpy arrays - now they should all be the same length
    #     for group in all_retention_rates:
    #         all_retention_rates[group] = np.array(all_retention_rates[group])
    #
    #     # Calculate statistics
    #     retention_stats = {}
    #     for group in all_retention_rates:
    #         rates = all_retention_rates[group]
    #         retention_stats[group] = {
    #             'mean': np.nanmean(rates, axis=0),  # Use nanmean to handle any NaN values
    #             'std': np.nanstd(rates, axis=0),
    #             'ci_lower': np.nanpercentile(rates, 25, axis=0),
    #             'ci_upper': np.nanpercentile(rates, 75, axis=0)
    #         }
    #
    #     # Create plot
    #     plt.figure(figsize=(12, 6))
    #
    #     colors = {
    #         'Protected (Top 25%)': 'darkblue',
    #         'Protected (Bottom 75%)': 'lightblue',
    #         'Unprotected (Top 25%)': 'darkgreen',
    #         'Unprotected (Bottom 75%)': 'lightgreen'
    #     }
    #
    #     styles = {
    #         'Protected (Top 25%)': '-',
    #         'Protected (Bottom 75%)': '--',
    #         'Unprotected (Top 25%)': '-',
    #         'Unprotected (Bottom 75%)': '--'
    #     }
    #
    #     # Plot each group with confidence intervals
    #     for group_name in retention_stats:
    #         stats = retention_stats[group_name]
    #         timesteps = range(len(stats['mean']))
    #
    #         # Plot mean line
    #         plt.plot(
    #             timesteps,
    #             stats['mean'],
    #             label=group_name,
    #             color=colors[group_name],
    #             linestyle=styles[group_name],
    #             linewidth=2
    #         )
    #
    #         # Add confidence interval
    #         plt.fill_between(
    #             timesteps,
    #             stats['ci_lower'],
    #             stats['ci_upper'],
    #             color=colors[group_name],
    #             alpha=0.2
    #         )
    #
    #     plt.xlabel('Time Step')
    #     plt.ylabel('Fraction Active')
    #     plt.title(f'Creator Retention by Initial Quality\n(Averaged over {len(self.all_results)} simulations)')
    #     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #     plt.grid(True, alpha=0.3)
    #
    #     # Add summary statistics
    #     stats_text = []
    #     for group_name, stats in retention_stats.items():
    #         final_mean = stats['mean'][-1]
    #         final_std = stats['std'][-1]
    #         stats_text.append(f"{group_name}:")
    #         stats_text.append(f"  Final: {final_mean:.1%} ± {final_std:.1%}")
    #
    #     plt.figtext(1.15, 0.5, '\n'.join(stats_text),
    #                 bbox=dict(facecolor='white', alpha=0.8))
    #
    #     plt.tight_layout()
    #
    #     if save_path:
    #         plt.savefig(save_path, bbox_inches='tight', dpi=300)
    #     plt.show()
    #
    #     # Print detailed statistics
    #     print("\nDetailed Retention Statistics:")
    #     print("-" * 40)
    #     for group_name, stats in retention_stats.items():
    #         print(f"\n{group_name}:")
    #         print(f"Final retention rate: {stats['mean'][-1]:.1%} ± {stats['std'][-1]:.1%}")
    #         print(f"Average retention rate: {np.mean(stats['mean']):.1%}")
    #
    #         # Time to 50% dropout (across all simulations)
    #         half_life_times = []
    #         for sim_rates in all_retention_rates[group_name]:
    #             half_life = next((i for i, r in enumerate(sim_rates) if r <= 0.5), None)
    #             if half_life is not None:
    #                 half_life_times.append(half_life)
    #
    #         if half_life_times:
    #             mean_half_life = np.mean(half_life_times)
    #             std_half_life = np.std(half_life_times)
    #             print(f"Time to 50% retention: {mean_half_life:.1f} ± {std_half_life:.1f} timesteps")

    def plot_retention(self, save_path: str = None):
        """
        Plot creator retention over time split into 4 groups across multiple simulations.
        Instead of truncating to the minimum number of timesteps, we use the maximum number
        and fill missing iterations (from early-terminated simulations) with NaN.
        """
        # Determine the maximum number of timesteps across all simulations.
        max_timesteps = 0
        for sim_data in self.all_results:
            active_status = np.array(sim_data['active_status_over_time'])
            max_timesteps = max(max_timesteps, len(active_status))

        # Initialize storage for retention rates across all simulations
        all_retention_rates = {
            'Protected (Top 25%)': [],
            'Protected (Bottom 75%)': [],
            'Unprotected (Top 25%)': [],
            'Unprotected (Bottom 75%)': []
        }

        # Process each simulation
        for sim_data in self.all_results:
            # Convert data to numpy arrays.
            # Note: Instead of truncating to min_timesteps, we record available data and pad the rest.
            active_status = np.array(sim_data['active_status_over_time'])
            quality_over_time = np.array(sim_data['quality_over_time'])
            is_protected = np.array(sim_data['is_CC_protected'])

            # We'll assume that the retention (active_status) is recorded per timestep.
            # For simulations that ended early, we'll pad the data with NaN.
            sim_timesteps = len(active_status)
            if sim_timesteps < max_timesteps:
                pad_length = max_timesteps - sim_timesteps
                active_status = np.concatenate((active_status, np.full((pad_length, active_status.shape[1]), np.nan)), axis=0)

            # Use the initial quality to create group masks
            initial_qualities = quality_over_time[0]
            protected_mask = is_protected
            unprotected_mask = ~is_protected

            # Determine thresholds based on initial quality
            if np.any(protected_mask):
                protected_qualities = initial_qualities[protected_mask]
                protected_threshold = np.percentile(protected_qualities, 75)
            else:
                protected_threshold = float('inf')
            if np.any(unprotected_mask):
                unprotected_qualities = initial_qualities[unprotected_mask]
                unprotected_threshold = np.percentile(unprotected_qualities, 75)
            else:
                unprotected_threshold = float('inf')

            # Create subgroup masks based on initial qualities.
            protected_top_mask = protected_mask & (initial_qualities >= protected_threshold)
            protected_bottom_mask = protected_mask & (initial_qualities < protected_threshold)
            unprotected_top_mask = unprotected_mask & (initial_qualities >= unprotected_threshold)
            unprotected_bottom_mask = unprotected_mask & (initial_qualities < unprotected_threshold)

            # Calculate retention rates for this simulation
            sim_retention_rates = {
                'Protected (Top 25%)': [],
                'Protected (Bottom 75%)': [],
                'Unprotected (Top 25%)': [],
                'Unprotected (Bottom 75%)': []
            }

            for t in range(max_timesteps):
                # For each timestep, if data is missing (NaN), record NaN retention rates.
                current_active = active_status[t]
                if np.isnan(current_active).all():
                    # If all entries are NaN, append NaN for every group.
                    for group in sim_retention_rates:
                        sim_retention_rates[group].append(np.nan)
                    continue

                # Otherwise, compute retention rates.
                for group_name, mask in [
                    ('Protected (Top 25%)', protected_top_mask),
                    ('Protected (Bottom 75%)', protected_bottom_mask),
                    ('Unprotected (Top 25%)', unprotected_top_mask),
                    ('Unprotected (Bottom 75%)', unprotected_bottom_mask)
                ]:
                    if np.any(mask):
                        rate = np.mean(current_active[mask])
                    else:
                        rate = np.nan  # If no creators in this group, mark as NaN.
                    sim_retention_rates[group_name].append(rate)

            # Append this simulation's rates
            for group in all_retention_rates:
                all_retention_rates[group].append(sim_retention_rates[group])

        # Convert lists to numpy arrays. Each array now has shape (num_simulations, max_timesteps)
        for group in all_retention_rates:
            all_retention_rates[group] = np.array(all_retention_rates[group])

        # Calculate statistics using nan-aware functions.
        retention_stats = {}
        for group in all_retention_rates:
            rates = all_retention_rates[group]
            retention_stats[group] = {
                'mean': np.nanmean(rates, axis=0),
                'std': np.nanstd(rates, axis=0),
                'ci_lower': np.nanpercentile(rates, 25, axis=0),
                'ci_upper': np.nanpercentile(rates, 75, axis=0)
            }

        # Create the plot
        plt.figure(figsize=(12, 6))

        colors = {
            'Protected (Top 25%)': 'orange',
            'Protected (Bottom 75%)': '#FFDAB9',
            'Unprotected (Top 25%)': 'darkblue',
            'Unprotected (Bottom 75%)': 'lightblue'
        }

        styles = {
            'Protected (Top 25%)': '-',
            'Protected (Bottom 75%)': '--',
            'Unprotected (Top 25%)': '-',
            'Unprotected (Bottom 75%)': '--'
        }

        for group_name in retention_stats:
            stats = retention_stats[group_name]
            timesteps = range(len(stats['mean']))
            plt.plot(
                timesteps,
                stats['mean'],
                label=group_name,
                color=colors[group_name],
                linestyle=styles[group_name],
                linewidth=2
            )
            plt.fill_between(
                timesteps,
                stats['ci_lower'],
                stats['ci_upper'],
                color=colors[group_name],
                alpha=0.2
            )

        plt.xlabel('Time Step')
        plt.ylabel('Fraction Active')
        plt.title(f'Creator Retention by Initial Quality\n(Averaged over {len(self.all_results)} simulations)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        # Add summary statistics text
        stats_text = []
        for group_name, stats in retention_stats.items():
            final_mean = stats['mean'][-1]
            final_std = stats['std'][-1]
            stats_text.append(f"{group_name}: Final: {final_mean:.1%} ± {final_std:.1%}")

        plt.figtext(1.15, 0.5, "\n".join(stats_text),
                    bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

        # Print detailed statistics.
        print("\nDetailed Retention Statistics:")
        print("-" * 40)
        for group_name, stats in retention_stats.items():
            print(f"\n{group_name}:")
            print(f"Final retention rate: {stats['mean'][-1]:.1%} ± {stats['std'][-1]:.1%}")
            print(f"Average retention rate: {np.nanmean(stats['mean']):.1%}")

            # Calculate time to 50% retention (if available)
            half_life_times = []
            for sim_rates in all_retention_rates[group_name]:
                half_life = next((i for i, r in enumerate(sim_rates) if not np.isnan(r) and r <= 0.5), None)
                if half_life is not None:
                    half_life_times.append(half_life)
            if half_life_times:
                mean_half_life = np.mean(half_life_times)
                std_half_life = np.std(half_life_times)
                print(f"Time to 50% retention: {mean_half_life:.1f} ± {std_half_life:.1f} timesteps")

    def calculate_quality_stats(self, metrics):
        """Helper function to calculate statistics safely"""
        has_data = ~np.isnan(metrics).all(axis=0)

        stats = {}
        stats['mean'] = np.zeros(metrics.shape[1])
        stats['std'] = np.zeros(metrics.shape[1])
        stats['ci_lower'] = np.zeros(metrics.shape[1])
        stats['ci_upper'] = np.zeros(metrics.shape[1])

        for t in range(metrics.shape[1]):
            if has_data[t]:
                valid_data = metrics[:, t][~np.isnan(metrics[:, t])]
                if len(valid_data) > 0:
                    stats['mean'][t] = np.mean(valid_data)
                    stats['std'][t] = np.std(valid_data) if len(valid_data) > 1 else 0
                    stats['ci_lower'][t] = np.percentile(valid_data, 25) if len(valid_data) > 1 else stats['mean'][t]
                    stats['ci_upper'][t] = np.percentile(valid_data, 75) if len(valid_data) > 1 else stats['mean'][t]

        return stats

    # def plot_quality_evolution(self, save_path: str = None):
    #     """
    #     Plot quality evolution over time for active and dropped out creators
    #     across multiple simulations
    #     """
    #     # Find minimum number of timesteps
    #     min_timesteps = float('inf')
    #     for sim_data in self.all_results:
    #         quality_over_time = np.array(sim_data['quality_over_time'])
    #         min_timesteps = min(min_timesteps, len(quality_over_time))
    #
    #     # Initialize storage for quality metrics
    #     all_quality_metrics = {
    #         'Protected (Active)': [],
    #         'Protected (Dropout)': [],
    #         'Unprotected (Active)': [],
    #         'Unprotected (Dropout)': []
    #     }
    #
    #     # Process each simulation
    #     for sim_data in self.all_results:
    #         # Convert data to numpy arrays
    #         active_status = np.array(sim_data['active_status_over_time'])[:min_timesteps]
    #         quality_over_time = np.array(sim_data['quality_over_time'])[:min_timesteps]
    #         is_protected = np.array(sim_data['is_CC_protected'])
    #
    #         # Store metrics for this simulation
    #         sim_quality_metrics = {
    #             'Protected (Active)': np.zeros(min_timesteps),
    #             'Protected (Dropout)': np.zeros(min_timesteps),
    #             'Unprotected (Active)': np.zeros(min_timesteps),
    #             'Unprotected (Dropout)': np.zeros(min_timesteps)
    #         }
    #
    #         # Calculate metrics for each timestep
    #         for t in range(min_timesteps):
    #             active_mask = active_status[t]
    #             dropout_mask = ~active_mask
    #
    #             # Protected active
    #             mask = is_protected & active_mask
    #             if np.any(mask):
    #                 sim_quality_metrics['Protected (Active)'][t] = np.mean(quality_over_time[t][mask])
    #             else:
    #                 sim_quality_metrics['Protected (Active)'][t] = np.nan
    #
    #             # Protected dropout
    #             mask = is_protected & dropout_mask
    #             if np.any(mask):
    #                 sim_quality_metrics['Protected (Dropout)'][t] = np.mean(quality_over_time[t][mask])
    #             else:
    #                 sim_quality_metrics['Protected (Dropout)'][t] = np.nan
    #
    #             # Unprotected active
    #             mask = ~is_protected & active_mask
    #             if np.any(mask):
    #                 sim_quality_metrics['Unprotected (Active)'][t] = np.mean(quality_over_time[t][mask])
    #             else:
    #                 sim_quality_metrics['Unprotected (Active)'][t] = np.nan
    #
    #             # Unprotected dropout
    #             mask = ~is_protected & dropout_mask
    #             if np.any(mask):
    #                 sim_quality_metrics['Unprotected (Dropout)'][t] = np.mean(quality_over_time[t][mask])
    #             else:
    #                 sim_quality_metrics['Unprotected (Dropout)'][t] = np.nan
    #
    #         # Store metrics from this simulation
    #         for group in all_quality_metrics:
    #             all_quality_metrics[group].append(sim_quality_metrics[group])
    #
    #     # Convert lists to numpy arrays
    #     for group in all_quality_metrics:
    #         all_quality_metrics[group] = np.array(all_quality_metrics[group])
    #
    #     # Calculate statistics
    #     quality_stats = {}
    #     for group in all_quality_metrics:
    #         metrics = all_quality_metrics[group]
    #         quality_stats[group] = self.calculate_quality_stats(metrics)
    #
    #     # Create plot
    #     plt.figure(figsize=(12, 6))
    #
    #     colors = {
    #         'Protected (Active)': 'blue',
    #         'Protected (Dropout)': 'blue',
    #         'Unprotected (Active)': 'green',
    #         'Unprotected (Dropout)': 'green'
    #     }
    #
    #     styles = {
    #         'Protected (Active)': '-',
    #         'Protected (Dropout)': '--',
    #         'Unprotected (Active)': '-',
    #         'Unprotected (Dropout)': '--'
    #     }
    #
    #     alphas = {
    #         'Protected (Active)': 1.0,
    #         'Protected (Dropout)': 0.7,
    #         'Unprotected (Active)': 1.0,
    #         'Unprotected (Dropout)': 0.7
    #     }
    #
    #     # Plot each group
    #     for group_name in quality_stats:
    #         stats = quality_stats[group_name]
    #         timesteps = range(len(stats['mean']))
    #
    #         if 'Dropout' in group_name:
    #             # For dropout groups, only plot after first dropout occurs
    #             start_idx = np.where(stats['mean'] != 0)[0]
    #             if len(start_idx) > 0:
    #                 start_idx = start_idx[0]
    #                 plt.plot(
    #                     timesteps[start_idx:],
    #                     stats['mean'][start_idx:],
    #                     label=f'{group_name} (n={len(self.all_results)})',
    #                     color=colors[group_name],
    #                     linestyle=styles[group_name],
    #                     alpha=alphas[group_name],
    #                     linewidth=2
    #                 )
    #
    #                 plt.fill_between(
    #                     timesteps[start_idx:],
    #                     stats['ci_lower'][start_idx:],
    #                     stats['ci_upper'][start_idx:],
    #                     color=colors[group_name],
    #                     alpha=0.2
    #                 )
    #         else:
    #             # Plot active groups normally
    #             plt.plot(
    #                 timesteps,
    #                 stats['mean'],
    #                 label=f'{group_name} (n={len(self.all_results)})',
    #                 color=colors[group_name],
    #                 linestyle=styles[group_name],
    #                 alpha=alphas[group_name],
    #                 linewidth=2
    #             )
    #
    #             plt.fill_between(
    #                 timesteps,
    #                 stats['ci_lower'],
    #                 stats['ci_upper'],
    #                 color=colors[group_name],
    #                 alpha=0.2
    #             )
    #
    #     plt.xlabel('Time Step')
    #     plt.ylabel('Average Quality')
    #     plt.title(f'Quality Evolution Over Time\n(Averaged over {len(self.all_results)} simulations)')
    #     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #     plt.grid(True, alpha=0.3)
    #
    #     # Add summary statistics
    #     stats_text = []
    #     for group_name, stats in quality_stats.items():
    #         if np.any(stats['mean'] != 0):
    #             first_valid = np.where(stats['mean'] != 0)[0][0]
    #             last_valid = np.where(stats['mean'] != 0)[0][-1]
    #             stats_text.append(f"{group_name}:")
    #             stats_text.append(f"  Final: {stats['mean'][last_valid]:.2f} ± {stats['std'][last_valid]:.2f}")
    #
    #     plt.figtext(1.15, 0.5, '\n'.join(stats_text),
    #                 bbox=dict(facecolor='white', alpha=0.8))
    #
    #     plt.tight_layout()
    #
    #     if save_path:
    #         plt.savefig(save_path, bbox_inches='tight', dpi=300)
    #     plt.show()
    #
    #     # Print statistics
    #     print("\nQuality Evolution Statistics:")
    #     print("-" * 40)
    #     for group_name, stats in quality_stats.items():
    #         print(f"\n{group_name}:")
    #         if np.any(stats['mean'] != 0):
    #             first_valid = np.where(stats['mean'] != 0)[0][0]
    #             last_valid = np.where(stats['mean'] != 0)[0][-1]
    #             print(f"Initial quality: {stats['mean'][first_valid]:.2f} ± {stats['std'][first_valid]:.2f}")
    #             print(f"Final quality: {stats['mean'][last_valid]:.2f} ± {stats['std'][last_valid]:.2f}")
    #             print(f"Quality growth: {stats['mean'][last_valid] - stats['mean'][first_valid]:.2f}")
    #         else:
    #             print("No data available for this group")

    def plot_quality_evolution(self, save_path: str = None):
        """
        Plot quality evolution over time for active and dropped-out creators
        across multiple simulations. Instead of truncating to the minimum number
        of timesteps, we use the maximum number of timesteps and pad missing values with NaN.
        """
        # Determine the maximum number of timesteps among all simulations.
        max_timesteps = 0
        for sim_data in self.all_results:
            quality_over_time = np.array(sim_data['quality_over_time'])
            max_timesteps = max(max_timesteps, len(quality_over_time))

        # Initialize storage for quality metrics across all simulations.
        all_quality_metrics = {
            'Protected (Active)': [],
            'Protected (Dropout)': [],
            'Unprotected (Active)': [],
            'Unprotected (Dropout)': []
        }

        # Process each simulation.
        for sim_data in self.all_results:
            quality_over_time = np.array(sim_data['quality_over_time'])
            active_status = np.array(sim_data['active_status_over_time'])
            is_protected = np.array(sim_data['is_CC_protected'])

            # Pad quality_over_time and active_status if simulation ended early.
            current_timesteps = len(quality_over_time)
            if current_timesteps < max_timesteps:
                pad_length = max_timesteps - current_timesteps
                # Pad quality with NaN rows.
                pad_quality = np.full((pad_length, quality_over_time.shape[1]), np.nan)
                quality_over_time = np.concatenate((quality_over_time, pad_quality), axis=0)
                # Pad active status similarly.
                pad_active = np.full((pad_length, active_status.shape[1]), np.nan)
                active_status = np.concatenate((active_status, pad_active), axis=0)

            # Use the initial quality (first timestep) to classify groups.
            initial_qualities = quality_over_time[0]
            protected_mask = is_protected
            unprotected_mask = ~is_protected

            # Determine thresholds based on initial quality.
            if np.any(protected_mask):
                protected_qualities = initial_qualities[protected_mask]
                protected_threshold = np.percentile(protected_qualities, 75)
            else:
                protected_threshold = float('inf')

            if np.any(unprotected_mask):
                unprotected_qualities = initial_qualities[unprotected_mask]
                unprotected_threshold = np.percentile(unprotected_qualities, 75)
            else:
                unprotected_threshold = float('inf')

            # Create subgroup masks.
            protected_top_mask = protected_mask & (initial_qualities >= protected_threshold)
            protected_bottom_mask = protected_mask & (initial_qualities < protected_threshold)
            unprotected_top_mask = unprotected_mask & (initial_qualities >= unprotected_threshold)
            unprotected_bottom_mask = unprotected_mask & (initial_qualities < unprotected_threshold)

            # Initialize metrics for this simulation.
            sim_quality_metrics = {
                'Protected (Active)': np.empty(max_timesteps),
                'Protected (Dropout)': np.empty(max_timesteps),
                'Unprotected (Active)': np.empty(max_timesteps),
                'Unprotected (Dropout)': np.empty(max_timesteps)
            }
            # We'll fill missing data with NaN.
            for key in sim_quality_metrics:
                sim_quality_metrics[key].fill(np.nan)

            # Calculate metrics for each timestep.
            for t in range(max_timesteps):
                # If the entire row is NaN, then data is missing for this timestep.
                if np.isnan(quality_over_time[t]).all():
                    # Leave the values as NaN.
                    continue

                current_active = active_status[t]
                # For each group, compute the average quality.
                for group_name, mask in [
                    ('Protected (Active)', protected_mask & current_active),
                    ('Protected (Dropout)', protected_mask & ~current_active),
                    ('Unprotected (Active)', unprotected_mask & current_active),
                    ('Unprotected (Dropout)', unprotected_mask & ~current_active)
                ]:
                    if np.any(mask):
                        sim_quality_metrics[group_name][t] = np.nanmean(quality_over_time[t][mask])
                    else:
                        sim_quality_metrics[group_name][t] = np.nan  # or leave it as NaN

            # Store metrics from this simulation.
            for group in all_quality_metrics:
                all_quality_metrics[group].append(sim_quality_metrics[group])

        # Convert lists to numpy arrays (shape: [num_simulations, max_timesteps]).
        for group in all_quality_metrics:
            all_quality_metrics[group] = np.array(all_quality_metrics[group])

        # Calculate statistics (using a helper function that handles NaNs).
        quality_stats = {}
        for group in all_quality_metrics:
            metrics = all_quality_metrics[group]
            quality_stats[group] = self.calculate_quality_stats(metrics)

        # Create plot.
        plt.figure(figsize=(12, 6))

        colors = {
            'Protected (Active)': 'orange',
            'Protected (Dropout)': '#FFDAB9',
            'Unprotected (Active)': 'darkblue',
            'Unprotected (Dropout)': 'lightblue'
        }

        styles = {
            'Protected (Active)': '-',
            'Protected (Dropout)': '--',
            'Unprotected (Active)': '-',
            'Unprotected (Dropout)': '--'
        }

        alphas = {
            'Protected (Active)': 1.0,
            'Protected (Dropout)': 0.7,
            'Unprotected (Active)': 1.0,
            'Unprotected (Dropout)': 0.7
        }

        # Plot each group.
        for group_name in quality_stats:
            stats = quality_stats[group_name]
            timesteps = range(len(stats['mean']))
            if 'Dropout' in group_name:
                # For dropout groups, plot only after the first valid value.
                valid_indices = np.where(~np.isnan(stats['mean']))[0]
                if valid_indices.size > 0:
                    start_idx = valid_indices[0]
                    plt.plot(
                        list(timesteps)[start_idx:],
                        stats['mean'][start_idx:],
                        label=f'{group_name} (n={len(self.all_results)})',
                        color=colors[group_name],
                        linestyle=styles[group_name],
                        alpha=alphas[group_name],
                        linewidth=2
                    )
                    plt.fill_between(
                        list(timesteps)[start_idx:],
                        stats['ci_lower'][start_idx:],
                        stats['ci_upper'][start_idx:],
                        color=colors[group_name],
                        alpha=0.2
                    )
            else:
                plt.plot(
                    timesteps,
                    stats['mean'],
                    label=f'{group_name} (n={len(self.all_results)})',
                    color=colors[group_name],
                    linestyle=styles[group_name],
                    alpha=alphas[group_name],
                    linewidth=2
                )
                plt.fill_between(
                    timesteps,
                    stats['ci_lower'],
                    stats['ci_upper'],
                    color=colors[group_name],
                    alpha=0.2
                )

        plt.xlabel('Time Step')
        plt.ylabel('Average Quality')
        plt.title(f'Quality Evolution Over Time\n(Averaged over {len(self.all_results)} simulations)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        # Add summary statistics text.
        stats_text = []
        for group_name, stats in quality_stats.items():
            valid = ~np.isnan(stats['mean'])
            if np.any(valid):
                first_valid = np.where(valid)[0][0]
                last_valid = np.where(valid)[0][-1]
                stats_text.append(f"{group_name}:")
                stats_text.append(f"  Final: {stats['mean'][last_valid]:.2f} ± {stats['std'][last_valid]:.2f}")
        plt.figtext(1.15, 0.5, "\n".join(stats_text),
                    bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

        # Print detailed statistics.
        print("\nQuality Evolution Statistics:")
        print("-" * 40)
        for group_name, stats in quality_stats.items():
            print(f"\n{group_name}:")
            valid = ~np.isnan(stats['mean'])
            if np.any(valid):
                first_valid = np.where(valid)[0][0]
                last_valid = np.where(valid)[0][-1]
                print(f"Initial quality: {stats['mean'][first_valid]:.2f} ± {stats['std'][first_valid]:.2f}")
                print(f"Final quality: {stats['mean'][last_valid]:.2f} ± {stats['std'][last_valid]:.2f}")
                print(f"Quality growth: {stats['mean'][last_valid] - stats['mean'][first_valid]:.2f}")
            else:
                print("No data available for this group")

    def calculate_activity_stats(self,metrics):
        """Helper function to calculate statistics safely"""
        has_data = ~np.isnan(metrics).all(axis=0)

        stats = {}
        stats['mean'] = np.zeros(metrics.shape[1])
        stats['std'] = np.zeros(metrics.shape[1])
        stats['ci_lower'] = np.zeros(metrics.shape[1])
        stats['ci_upper'] = np.zeros(metrics.shape[1])

        for t in range(metrics.shape[1]):
            if has_data[t]:
                valid_data = metrics[:, t][~np.isnan(metrics[:, t])]
                if len(valid_data) > 0:
                    stats['mean'][t] = np.mean(valid_data)
                    stats['std'][t] = np.std(valid_data) if len(valid_data) > 1 else 0
                    stats['ci_lower'][t] = np.percentile(valid_data, 25) if len(valid_data) > 1 else stats['mean'][t]
                    stats['ci_upper'][t] = np.percentile(valid_data, 75) if len(valid_data) > 1 else stats['mean'][t]

        return stats

    def plot_activity_scores(self, save_path: str = None):
        """
        Plot activity scores over time for multiple simulations, split by protected status
        and activity level. This version uses the maximum number of timesteps across simulations.
        For simulations that ended early, missing timesteps are padded with NaN, ensuring that
        all iterations are shown on the graph.
        """
        # Determine the maximum number of timesteps across all simulations.
        max_timesteps = 0
        for sim_data in self.all_results:
            ts = len(sim_data['active_status_over_time'])
            max_timesteps = max(max_timesteps, ts)

        # Initialize storage for activity metrics for each simulation.
        all_activity_metrics = {
            'Protected (High Activity)': [],
            'Protected (Low Activity)': [],
            'Unprotected (High Activity)': [],
            'Unprotected (Low Activity)': []
        }

        # Process each simulation.
        for sim_data in self.all_results:
            # Retrieve parameters.
            alpha = sim_data.get('alpha_activity', 1.0)
            beta = sim_data.get('beta_activity', 2.0)

            # Get active_status data and pad to max_timesteps if needed.
            active_status = np.array(sim_data['active_status_over_time'])
            if active_status.shape[0] < max_timesteps:
                pad_rows = max_timesteps - active_status.shape[0]
                pad_active = np.full((pad_rows, active_status.shape[1]), np.nan)
                active_status = np.concatenate([active_status, pad_active], axis=0)

            # Get is_CC_protected (assumed to be a 1D array; same for all timesteps).
            is_protected = np.array(sim_data['is_CC_protected'])

            # Get posts_recent, followers_current, total_followers and pad if needed.
            posts_recent = np.array(sim_data['posts_recent'])
            if posts_recent.shape[0] < max_timesteps:
                pad_rows = max_timesteps - posts_recent.shape[0]
                pad_posts = np.full((pad_rows, posts_recent.shape[1]), np.nan)
                posts_recent = np.concatenate([posts_recent, pad_posts], axis=0)

            followers_current = np.array(sim_data['followers_current'])
            if followers_current.shape[0] < max_timesteps:
                pad_rows = max_timesteps - followers_current.shape[0]
                pad_followers = np.full((pad_rows, followers_current.shape[1]), np.nan)
                followers_current = np.concatenate([followers_current, pad_followers], axis=0)

            total_followers = np.array(sim_data['total_followers'])
            if total_followers.shape[0] < max_timesteps:
                pad_rows = max_timesteps - total_followers.shape[0]
                pad_total = np.full((pad_rows, total_followers.shape[1]), np.nan)
                total_followers = np.concatenate([total_followers, pad_total], axis=0)

            # Initialize arrays for this simulation.
            sim_activity_metrics = {
                'Protected (High Activity)': np.full(max_timesteps, np.nan),
                'Protected (Low Activity)': np.full(max_timesteps, np.nan),
                'Unprotected (High Activity)': np.full(max_timesteps, np.nan),
                'Unprotected (Low Activity)': np.full(max_timesteps, np.nan)
            }

            # Process each timestep.
            for t in range(max_timesteps):
                # If no data is available at this timestep, skip.
                if np.isnan(posts_recent[t]).all():
                    continue

                # Compute activity scores.
                p = posts_recent[t]
                curr_f = followers_current[t]
                tot_f = total_followers[t]
                engagement_rate = curr_f / np.maximum(tot_f, 1)
                activity_scores = alpha * p + beta * engagement_rate

                # Use active_status at timestep t.
                current_active = active_status[t]
                # Compute scores for active CCs in each group.
                protected_scores = activity_scores[is_protected & current_active]
                unprotected_scores = activity_scores[~is_protected & current_active]

                # For protected group: use median of active scores as threshold.
                if len(protected_scores) > 0:
                    prot_thresh = np.median(protected_scores)
                    mask_high = is_protected & current_active & (activity_scores >= prot_thresh)
                    mask_low = is_protected & current_active & (activity_scores < prot_thresh)
                    if np.any(mask_high):
                        sim_activity_metrics['Protected (High Activity)'][t] = np.nanmean(activity_scores[mask_high])
                    if np.any(mask_low):
                        sim_activity_metrics['Protected (Low Activity)'][t] = np.nanmean(activity_scores[mask_low])

                # For unprotected group: use median of active scores as threshold.
                if len(unprotected_scores) > 0:
                    unprot_thresh = np.median(unprotected_scores)
                    mask_high = ~is_protected & current_active & (activity_scores >= unprot_thresh)
                    mask_low = ~is_protected & current_active & (activity_scores < unprot_thresh)
                    if np.any(mask_high):
                        sim_activity_metrics['Unprotected (High Activity)'][t] = np.nanmean(activity_scores[mask_high])
                    if np.any(mask_low):
                        sim_activity_metrics['Unprotected (Low Activity)'][t] = np.nanmean(activity_scores[mask_low])

            # Append this simulation's metrics.
            for group in all_activity_metrics:
                all_activity_metrics[group].append(sim_activity_metrics[group])

        # Convert lists to numpy arrays (shape: [num_simulations, max_timesteps]).
        for group in all_activity_metrics:
            all_activity_metrics[group] = np.array(all_activity_metrics[group])

        # Calculate statistics using a nan-aware helper function.
        activity_stats = {}
        for group in all_activity_metrics:
            metrics = all_activity_metrics[group]
            activity_stats[group] = self.calculate_activity_stats(metrics)

        # Create the plot.
        plt.figure(figsize=(12, 6))
        colors = {
            'Protected (High Activity)': 'orange',
            'Protected (Low Activity)': '#FFDAB9',
            'Unprotected (High Activity)': 'darkblue',
            'Unprotected (Low Activity)': 'lightblue'
        }
        styles = {
            'Protected (High Activity)': '-',
            'Protected (Low Activity)': '--',
            'Unprotected (High Activity)': '-',
            'Unprotected (Low Activity)': '--'
        }

        timesteps = range(max_timesteps)
        for group_name in activity_stats:
            stats = activity_stats[group_name]
            plt.plot(
                timesteps,
                stats['mean'],
                label=f'{group_name} (n={len(self.all_results)})',
                color=colors[group_name],
                linestyle=styles[group_name],
                linewidth=2
            )
            plt.fill_between(
                timesteps,
                stats['ci_lower'],
                stats['ci_upper'],
                color=colors[group_name],
                alpha=0.2
            )

        plt.xlabel('Time Step')
        plt.ylabel('Average Activity Score')
        plt.title(f'Activity Scores Over Time\n(Averaged over {len(self.all_results)} simulations)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        # Add summary statistics text.
        stats_text = []
        for group_name, stats in activity_stats.items():
            valid = np.where(~np.isnan(stats['mean']))[0]
            if valid.size > 0:
                last_valid = valid[-1]
                stats_text.append(
                    f"{group_name}: Final: {stats['mean'][last_valid]:.2f} ± {stats['std'][last_valid]:.2f}")

        plt.figtext(1.15, 0.5, "\n".join(stats_text),
                    bbox=dict(facecolor='white', alpha=0.8))

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

        # Print detailed statistics.
        print("\nActivity Score Statistics:")
        print("-" * 40)
        for group_name, stats in activity_stats.items():
            print(f"\n{group_name}:")
            valid_indices = np.where(~np.isnan(stats['mean']))[0]
            if valid_indices.size > 0:
                first_valid = valid_indices[0]
                last_valid = valid_indices[-1]
                print(f"Initial activity: {stats['mean'][first_valid]:.2f} ± {stats['std'][first_valid]:.2f}")
                print(f"Final activity: {stats['mean'][last_valid]:.2f} ± {stats['std'][last_valid]:.2f}")
                print(f"Peak activity: {np.nanmax(stats['mean']):.2f}")
                print(f"Average activity: {np.nanmean(stats['mean'][~np.isnan(stats['mean'])]):.2f}")
            else:
                print("No data available for this group")

    def plot_time_to_dropout(self, save_path: str = None):
        """
        Plot time-to-dropout distributions using survival curves.

        For each simulation in self.all_results, we extract:
          - active_status_over_time (2D array: iterations x num_creators)
          - is_CC_protected (1D array: length=num_creators)

        For each creator, the dropout time is defined as the first iteration where they become inactive.
        If they never drop out, we record the final timestep and mark them as censored.

        Survival curves are then plotted separately for protected and unprotected groups.
        """
        # import pandas as pd
        # import matplotlib.pyplot as plt

        # Initialize lists to collect data across all simulations.
        all_dropout_times = []
        all_is_censored = []
        all_protected_status = []

        # Loop over each simulation result.
        for sim_data in self.all_results:
            # Extract the active status array and the protection flag.
            # active_status_over_time is assumed to be a 2D array (iterations x num_creators)
            active_status = np.array(sim_data['active_status_over_time'])
            # is_CC_protected is a 1D array (length=num_creators)
            is_prot = np.array(sim_data['is_CC_protected'])
            num_creators = len(is_prot)

            # For each creator in the simulation, determine dropout time.
            for i in range(num_creators):
                creator_status = active_status[:, i]  # All timesteps for creator i
                # Find the first timestep where the creator is inactive.
                dropout_idx = np.where(~creator_status)[0]
                if dropout_idx.size > 0:
                    all_dropout_times.append(dropout_idx[0])
                    all_is_censored.append(False)
                else:
                    all_dropout_times.append(len(creator_status))
                    all_is_censored.append(True)
                all_protected_status.append(is_prot[i])

        # Create a DataFrame for survival analysis.
        df = pd.DataFrame({
            'time': all_dropout_times,
            'censored': all_is_censored,
            'protected': all_protected_status
        })

        # Plot survival curves.
        plt.figure(figsize=(10, 6))
        for prot in [True, False]:
            group_df = df[df['protected'] == prot]
            # Get unique time values (sorted).
            times = sorted(group_df['time'].unique())
            survival = []
            for t in times:
                # Survival probability: fraction of creators with dropout time >= t.
                survival.append(np.mean(group_df['time'] >= t))
            label = 'Protected' if prot else 'Unprotected'
            color = 'orange' if prot else 'blue'
            plt.step(times, survival, where='post', label=label, color=color)

        plt.xlabel('Time Step')
        plt.ylabel('Survival Probability')
        plt.title('Creator Survival Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()


# Example usage:
if __name__ == "__main__":
    # Initialize visualizer with results file
    viz = SimulationVisualizer('Simulation_results/')

    # Create all plots
    viz.plot_retention('retention.png')
    viz.plot_quality_evolution('quality.png')
    viz.plot_activity_scores('activity.png')
    viz.plot_time_to_dropout('dropout.png')