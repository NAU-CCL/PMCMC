##################
#pipeline for wrong lambda using PF

rm run_qsub_all_no_d.sh

date=$(date '+%Y-%m-%d')
sed '$d' pmcmc_job_v01.sh > temp_1st.sh

for (( counter_T=0; counter_T<15; counter_T++ ))
do
    # Create a new job script for each counter_T value
    job_file="${date}_dcorrelation_time_T${counter_T}.sh"
    > "$job_file"  # Clear or create the job file
    cat temp_1st.sh >> "$job_file"

    for (( counter_W=0; counter_W<15; counter_W++ ))
    do
        for (( run=0; run<10; run++ ))
        do
            echo "python experiment_script_SIHR_wrongT_PF_BM.py $counter_T $counter_W $run" >> "$job_file"
        done
    done

    # Append the qsub command for each job file to the main submission script
    echo "qsub $job_file" >> run_qsub_all_no_d.sh
done