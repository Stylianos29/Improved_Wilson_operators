#!/bin/bash -l
#SBATCH --job-name=CritMass
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=64
#SBATCH --time=6:00:00
#SBATCH --partition=p100
#SBATCH --error=./jobs_info/master_run.err
#SBATCH --output=./jobs_info/master_run.txt

module purge
module load gompi
module load Anaconda3
module load h5py

FERMION_DIMENSIONS=4

total_number_of_configurations=100
max_number_of_processses=64

configurations_per_iteration=$((max_number_of_processses/FERMION_DIMENSIONS))
number_of_iterations=$((total_number_of_configurations/configurations_per_iteration))
configurations_for_the_final_iteration=$((total_number_of_configurations%configurations_per_iteration))

for ((bare_mass_value = 1 ; bare_mass_value < 6 ; bare_mass_value++)); do

  counter=1
  for ((initial_index = 20 ; initial_index < 2001 ; initial_index+=$((configurations_per_iteration*20)))); do

    echo Iteration: $counter Bare mass value '0.0'"$bare_mass_value"

    if [[ $counter -le $number_of_iterations ]]; then
      sbatch --error='./jobs_info/WillStan'"$counter"'.err' --output='./jobs_info/WillStan'"$counter"'.txt' run.sh $max_number_of_processses $initial_index '0.0'"$bare_mass_value"
    else
      # Final iteration
      sbatch --error='./jobs_info/WillStan'"$counter"'.err' --output='./jobs_info/WillStan'"$counter"'.txt' run.sh $((configurations_for_the_final_iteration*FERMION_DIMENSIONS)) $initial_index '0.0'"$bare_mass_value"
    fi

    counter=$((counter+1))
  done

done
