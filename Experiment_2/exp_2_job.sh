####################
#job script for PF BM experiments with diff lambda

rm run_qsub_all_no_d.sh

date=$(date '+%Y-%m-%d')
sed '$d' pmcmc_job_v01.sh > temp_1st.sh

for (( counter=0; counter<15; counter++ ))
do
    # Create a temporary file for the current counter
    temp_file="${date}_dcorrelation_time_${counter}.sh"
    
    for (( run=0; run<50; run++ ))
    do
        echo "python experiment_script_SIHR_BM_v05.py $counter $run" >> "$temp_file"
    done
    
    # Combine temp_1st.sh with the current temp_file to create the final job file
    cat temp_1st.sh "$temp_file" > ${date}_dcorrelation_time_${counter}_complete.sh
    echo "qsub ${date}_dcorrelation_time_${counter}_complete.sh" >> run_qsub_all_no_d.sh
    
    # Remove the temporary counter file
    rm "$temp_file"
done