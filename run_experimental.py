import os
import json
import numpy as np

# Adjust these imports according to your module/file structure
import experimental_setup as exp  # Must include configs_dynamic_quality or generate_config_csvs
import simulation as sim  # Must include the updated dynamic simulation logic


def run_test():
    """
    Demonstrates how to generate multiple configurations with dynamic-quality parameters,
    run simulations, and analyze the results.
    This assumes you have integrated your dynamic-quality logic into 'model.py' and 'simulation.py'.
    """
    # 1) Define a base config with dynamic parameters
    #    (some keys here may be placeholders unless implemented in your dynamic code)
    base_config = {
        'num_users': 1000,  # Demo: smaller scale for local testing
        'num_items': 50,
        'type_attributes': 'multidimensional',
        'num_attributes': '"[1, 1]"',
        'kind_attributes': '''"['c', 'm']"''',
        'dict_cov': '"{(0,0):1, (1,1):1, (0,1):0, (1,0):0}"',
        'cumulative_weights': '[-1]',
        'matching_bound': 0,
        'prob_cumulative_weights': '[1]',

        # The number of steps you want to iterate before stopping or hitting convergence
        'num_steps': 100,

        # Example random seed
        'random_seed': 42,

        # DYNAMIC-QUALITY SPECIFIC (if you've integrated them in your code):
        'learning_rate': 0.01,
        'fatigue_lambda': 0.0005,
        'dropout_threshold': 0.3,
        'alpha_activity': 6.0,
        'beta_activity': 7.0,
        'prob_post': 0.7,

        # Additional placeholders that you might want to handle in your code
        'dropout_bias': 1.2,  # E.g. additional bias factor for protected creators
        'stagnation_penalty': 5,  # E.g. penalize creators who don't grow
        'quality_penalty': 35,  # E.g. penalize creators below some quality
        'growth_penalty': 35,  # E.g. penalize if growth is negative
        'min_followers_threshold': 1,  # E.g. require at least X followers to remain active
        'stagnation_threshold': 2  # E.g. how many steps of no growth => "stagnation"
    }

    # 2) Define how certain parameters change across runs
    #    For example, we vary the RS model, group size, and random seeds.
    #    The code below will generate a set of config files, each with a unique combination.

    changing = {}
    var_lists = []

    # (a) Recommendation system models
    rs_models = ['PA', 'UR', 'ExtremePA', 'biased_PA', 'biased_UR']
    # rs_models = ['PA']
    changing[0] = [[model] for model in rs_models]
    var_lists.append(['rs_model'])

    # (b) Different group sizes
    group_sizes = [0.1, 0.25, 0.5, 0.75]
    # group_sizes = [0.1]
    changing[1] = [[size] for size in group_sizes]
    var_lists.append(['%_groupA'])  # name used by your 'generate_CCs' logic for minority group size

    # (c) Random seeds (ensuring multiple runs)
    np.random.seed(42)  # so seeds are reproducible
    seeds = np.random.randint(0, 1000000, size=50)
    changing[2] = [[seed] for seed in seeds]
    var_lists.append(['random_seed'])

    # 3) Generate CSV config files
    #    All files go into 'Simulation_results' folder, and we put them into 1 partition (no_folders=1).
    print("Generating configuration files...")
    exp.generate_config_csvs(
        base_config=base_config,
        changing=changing,
        var_lists=var_lists,
        path='Simulation_results',
        no_folders=1  # single text file listing all configs
    )

    # 4) Run the generated configurations
    #    This looks for the text file 'file_names1.txt' in 'Simulation_results',
    #    which was created by generate_config_csvs, and runs each config in a loop.
    print("\nRunning simulations...")
    sim.run_sims('Simulation_results/file_names1.txt')

    # 5) Print results from each 'runX.json'
    print("\nAnalyzing results:")
    for filename in os.listdir('Simulation_results'):
        if filename.startswith('run') and filename.endswith('.json'):
            filepath = os.path.join('Simulation_results', filename)
            with open(filepath, 'r') as f:
                result = json.load(f)

            print(f"\nResults for {filename}:")

            # Print some config / result details (filter out large arrays like G, if you want)
            config_keys_to_ignore = {'num_followers', 'num_followees', 'G', 'is_CC_protected',
                                     'active_status_over_time', 'quality_over_time', 'dropout_history'}

            # Show config or run fields that aren't huge
            for key, val in result.items():
                if key not in config_keys_to_ignore:
                    print(f"{key}: {val}")

            # Optionally analyze followers
            if 'num_followers' in result and 'is_CC_protected' in result:
                print("\nAnalysis:")
                protected_flags = result['is_CC_protected']
                protected_count = sum(protected_flags)
                unprotected_count = len(protected_flags) - protected_count

                print(f"Protected creators: {protected_count}")
                print(f"Unprotected creators: {unprotected_count}")

                protected_followers = [
                    f for f, p in zip(result['num_followers'], protected_flags) if p
                ]
                unprotected_followers = [
                    f for f, p in zip(result['num_followers'], protected_flags) if not p
                ]

                if protected_followers:
                    avg_protected = sum(protected_followers) / len(protected_followers)
                    print(f"Avg followers (protected): {avg_protected:.2f}")
                if unprotected_followers:
                    avg_unprotected = sum(unprotected_followers) / len(unprotected_followers)
                    print(f"Avg followers (unprotected): {avg_unprotected:.2f}")

            # If you integrated dynamic-quality logging:
            if 'quality_over_time' in result:
                # Example: print only the final iteration's average CC quality
                final_iteration_qualities = result['quality_over_time'][-1]  # last iteration
                if final_iteration_qualities:
                    avg_quality_final = sum(final_iteration_qualities) / len(final_iteration_qualities)
                    print(f"Final avg CC quality: {avg_quality_final:.2f}")

            print("---")


def analyze_simulation_results(result):
    """Detailed analysis of a single simulation run,
       focusing on protected vs unprotected creators' statistics."""
    protected = sum(1 for is_protected in result['is_CC_protected'] if is_protected)
    unprotected = len(result['is_CC_protected']) - protected

    protected_followers = [f for f, p in zip(result['num_followers'], result['is_CC_protected']) if p]
    unprotected_followers = [f for f, p in zip(result['num_followers'], result['is_CC_protected']) if not p]

    stats = {
        "creator_counts": {
            "protected": protected,
            "unprotected": unprotected,
            "protected_percentage": (protected / (protected + unprotected)) * 100 if protected + unprotected > 0 else 0
        },
        "follower_stats": {
            "protected": {
                "avg": np.mean(protected_followers) if protected_followers else 0,
                "max": max(protected_followers) if protected_followers else 0,
                "min": min(protected_followers) if protected_followers else 0
            },
            "unprotected": {
                "avg": np.mean(unprotected_followers) if unprotected_followers else 0,
                "max": max(unprotected_followers) if unprotected_followers else 0,
                "min": min(unprotected_followers) if unprotected_followers else 0
            }
        }
    }

    print("\n[Detailed Analysis]")
    print("Creator Distribution:")
    print(f"- Protected: {stats['creator_counts']['protected']} "
          f"({stats['creator_counts']['protected_percentage']:.1f}%)")
    print(f"- Unprotected: {stats['creator_counts']['unprotected']}")

    print("\nFollower Statistics:")
    print("Protected Creators:")
    print(f"- Average: {stats['follower_stats']['protected']['avg']:.2f}")
    print(f"- Max: {stats['follower_stats']['protected']['max']}")
    print(f"- Min: {stats['follower_stats']['protected']['min']}")

    print("\nUnprotected Creators:")
    print(f"- Average: {stats['follower_stats']['unprotected']['avg']:.2f}")
    print(f"- Max: {stats['follower_stats']['unprotected']['max']}")
    print(f"- Min: {stats['follower_stats']['unprotected']['min']}")

    return stats


if __name__ == "__main__":
    """
    Run the test if this file is executed directly. 
    Ensure 'Simulation_results' directory exists,
    then generate config files, run them, and print aggregated results.
    """
    if not os.path.exists('Simulation_results'):
        os.makedirs('Simulation_results')
    run_test()
