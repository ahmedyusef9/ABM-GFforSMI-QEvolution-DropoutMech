
# run_experiment.py

This script serves as the main driver for our simulation experiments. It generates configuration files, runs simulations based on those configurations, and then performs basic analysis of the results.

## Overview

- **Configuration Generation:** 
  The script defines a base configuration with dynamic quality parameters (e.g., learning rate, fatigue lambda, dropout threshold) and variable parameters (such as recommendation system models, minority group proportions, and random seeds). These configurations are saved as CSV files.

- **Simulation Execution:**  
  Once the configuration files are generated, the script runs the simulations using these files. The simulations mimic a digital ecosystem with 1000 users and 50 content creators over 100 iterations.

- **Result Analysis:**  
  After running the simulations, the script reads the resulting JSON files, extracts key metrics (like retention rates, quality evolution, and activity scores), and prints summary statistics for further discussion.

## How to Run

1. **Dependencies:**  
   Make sure you have Python 3.x installed along with the necessary packages. You can install the required packages using:
   ```bash
   pip install numpy pandas matplotlib seaborn
   ```

2. **Run the Script:**  
   Execute the script from the command line:
   ```bash
   python run_experiment.py
   ```

3. **Output:**  
   - Configuration CSV files will be generated in the `Simulation_results` folder.
   - Simulation results will be saved as JSON files in the same folder.
   - The script will print summary statistics to the console for further analysis.

## Notes

- Ensure that the required folders (e.g., `Simulation_results`) exist or are created by the script.
- Adjust configuration parameters in the script as needed to fit your experimental setup.

```