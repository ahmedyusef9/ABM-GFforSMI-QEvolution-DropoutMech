import os
import json
import numpy as np
import matplotlib.pyplot as plt


class RetentionPlotter:
    """
    A class that loads multiple simulation runs (each as a JSON file),
    computes the fraction of active creators over time for protected vs. unprotected groups,
    and plots the aggregated retention curves.

    Each JSON is expected to have at least:
      - 'quality_over_time': 2D list [iteration][cc_id] (not used here, but typical of your format).
      - 'is_CC_protected': 1D bool list, len=#CCs, indicating if each CC is in the protected group.
      - 'active_status_over_time': 2D bool list [iteration][cc_id],
        indicating if each CC is active at each iteration.
      - possibly other fields, but the ones above are crucial for computing fraction of active CCs.
    """

    def __init__(self, results_dir: str):
        """
        Initialize by loading all runX.json files from a directory.

        :param results_dir: Folder path containing the .json simulation output files.
        """
        self.results_dir = results_dir
        self.runs = self._load_simulations()

    def _load_simulations(self):
        """
        Scans self.results_dir for files named run*.json, and loads them into a list of dicts.

        :return: A list of simulation dicts, each containing 'active_status_over_time'
                 and 'is_CC_protected' at minimum.
        """
        simulations = []
        for filename in os.listdir(self.results_dir):
            if filename.startswith("run") and filename.endswith(".json"):
                filepath = os.path.join(self.results_dir, filename)
                with open(filepath, 'r') as f:
                    sim_data = json.load(f)
                # Optionally store filename for reference
                sim_data['filename'] = filename
                simulations.append(sim_data)
        return simulations

    def compute_fraction_active(self):
        """
        Aggregates (across all runs) the fraction of active CCs at each iteration
        for protected vs. unprotected groups.

        Process:
          1) For each run:
             - We read 'active_status_over_time' = [iteration][cc_id].
             - We read 'is_CC_protected' = [cc_id].
             - For each iteration t:
               * Count how many protected CCs are active => p_active
               * Count how many unprotected CCs are active => up_active
               * Also note how many CCs are total in each group (use the first iteration or entire length).
               * We compute fraction_of_active_protected[t] = p_active / total_protected
                 and similarly for unprotected. (We define total_protected as # of protected CCs
                 at the start, so it's truly a retention fraction.)
          2) Because different runs might have different #iterations,
             we track the maximum iteration length across runs, then combine them.
          3) We produce arrays p_mean[t], up_mean[t] (average fraction across runs),
             and possibly p_std[t], up_std[t] (standard deviation).

        :return: A dict with keys = 'timesteps', 'p_mean', 'p_std', 'up_mean', 'up_std'.
                 Where:
                   - timesteps is an integer range of length = max number of iterations among runs
                   - p_mean[t], up_mean[t] are the average fraction active at iteration t
                   - p_std[t], up_std[t] are standard deviations across runs
        """
        # 1) Find max number of iterations
        max_iters = 0
        for run in self.runs:
            active_status = run['active_status_over_time']
            if len(active_status) > max_iters:
                max_iters = len(active_status)

        # We'll store for each iteration:
        # a list of fraction_active_protected from each run
        # a list of fraction_active_unprotected from each run
        p_fractions_runs = [[] for _ in range(max_iters)]
        up_fractions_runs = [[] for _ in range(max_iters)]

        for run in self.runs:
            active_status = run['active_status_over_time']  # shape [iteration][cc_id]
            is_protected = run['is_CC_protected']
            num_ccs = len(is_protected)

            # Count how many are protected vs unprotected in total
            total_protected = sum(is_protected)
            total_unprotected = num_ccs - total_protected

            # For each iteration t, count how many are active in each group
            for t in range(len(active_status)):
                # active_status[t] is a bool list of length num_ccs
                row = active_status[t]
                p_active = 0
                up_active = 0

                for i, (a, p) in enumerate(zip(row, is_protected)):
                    if a and p:
                        p_active += 1
                    elif a and not p:
                        up_active += 1

                # Fraction of the original group that remains active
                if total_protected > 0:
                    p_fractions_runs[t].append(p_active / total_protected)
                else:
                    p_fractions_runs[t].append(np.nan)

                if total_unprotected > 0:
                    up_fractions_runs[t].append(up_active / total_unprotected)
                else:
                    up_fractions_runs[t].append(np.nan)

            # if this run has fewer iters than max_iters,
            # fill the rest with np.nan
            for t in range(len(active_status), max_iters):
                p_fractions_runs[t].append(np.nan)
                up_fractions_runs[t].append(np.nan)

        # Now compute mean & std across runs for each iteration
        p_mean = []
        p_std = []
        up_mean = []
        up_std = []
        for t in range(max_iters):
            # ignore NaNs
            pvals = [v for v in p_fractions_runs[t] if not np.isnan(v)]
            upvals = [v for v in up_fractions_runs[t] if not np.isnan(v)]

            if pvals:
                p_mean.append(np.mean(pvals))
                p_std.append(np.std(pvals))
            else:
                p_mean.append(np.nan)
                p_std.append(np.nan)

            if upvals:
                up_mean.append(np.mean(upvals))
                up_std.append(np.std(upvals))
            else:
                up_mean.append(np.nan)
                up_std.append(np.nan)

        return {
            'timesteps': range(max_iters),
            'p_mean': p_mean,
            'p_std': p_std,
            'up_mean': up_mean,
            'up_std': up_std
        }

    def plot_fraction_active(self, confidence_bands=True):
        """
        Produces a plot where:
          - x_axis = iteration (timesteps)
          - y_axis = fraction of active creators
          - lines = Protected group vs. Unprotected group,
            aggregated across all runs.

        Args:
            confidence_bands (bool): If True, fill ±1 std dev around the mean lines
                                     to show variability across runs.

        The final figure addresses "Creator Retention Over Time," i.e.:
          "At iteration t, what fraction of the original group is still active?"
          (for both protected vs unprotected).

        If your dataset is large or you have many runs, this line plot with
        optional confidence shading is a concise way to see whether protected
        CCs systematically drop out faster or remain at similar levels.
        """
        data = self.compute_fraction_active()
        x_vals = data['timesteps']

        p_mean = data['p_mean']
        p_std = data['p_std']
        up_mean = data['up_mean']
        up_std = data['up_std']

        plt.figure(figsize=(10, 6))
        # Protected line
        plt.plot(x_vals, p_mean, label='Protected Group', color='red')
        if confidence_bands:
            pm_lower = [m - s if not np.isnan(m) else np.nan for m, s in zip(p_mean, p_std)]
            pm_upper = [m + s if not np.isnan(m) else np.nan for m, s in zip(p_mean, p_std)]
            plt.fill_between(x_vals, pm_lower, pm_upper, alpha=0.2, color='red')

        # Unprotected line
        plt.plot(x_vals, up_mean, label='Unprotected Group', color='blue', linestyle='--')
        if confidence_bands:
            upm_lower = [m - s if not np.isnan(m) else np.nan for m, s in zip(up_mean, up_std)]
            upm_upper = [m + s if not np.isnan(m) else np.nan for m, s in zip(up_mean, up_std)]
            plt.fill_between(x_vals, upm_lower, upm_upper, alpha=0.2, color='blue')

        plt.title("Creator Retention Over Time")
        plt.xlabel("Timesteps")
        plt.ylabel("Fraction of Active Creators")
        plt.legend()
        plt.grid(True)
        plt.show()


# Example usage if you run this script directly
if __name__ == "__main__":
    """
    Suppose you have a directory 'Simulation_results' with run0.json, run1.json, etc.
    Each file has:
      - 'active_status_over_time': [ [bool,...], [bool,...], ... ]
      - 'is_CC_protected': [bool, bool, ...]
    among other fields.

    The code below:
      1) Instantiates RetentionPlotter with that directory
      2) Aggregates fraction of active CCs for protected vs. unprotected
      3) Plots the aggregated line + standard deviation
    """
    rp = RetentionPlotter("Simulation_results")
    rp.plot_fraction_active(confidence_bands=True)
