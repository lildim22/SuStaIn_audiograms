#!/bin/bash

# Define the log file location
LOGFILE="nhanes.txt"

# List of Python scripts to run
scripts=("nhanes_1000_run.py", "nhanes_1000_mcmc_trace.py", "nhanes_1000_.py")

# Function to run a Python script and log its execution time
run_script() {
    local script_name=$1

    # Log start time
    echo "Starting $script_name at $(date)" >> $LOGFILE

    # Record start time in seconds
    local start_time=$(date +%s)

    # Execute the Python script and redirect output to a separate log file
    python3 $script_name > "${script_name%}_$(date +%Y%m%d_%H%M%S).log" 2>&1

    # Record end time in seconds
    local end_time=$(date +%s)

    # Calculate duration
    local duration=$((end_time - start_time))

    # Log end time and duration
    echo "Completed $script_name at $(date)" >> $LOGFILE
    echo "Duration of $script_name: $duration seconds" >> $LOGFILE
    echo "---------------------------------" >> $LOGFILE
}

# Loop through the list of scripts and run them
for script in "${scripts[@]}"; do
    run_script $script
done

echo "All scripts executed successfully."