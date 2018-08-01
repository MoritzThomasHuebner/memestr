#!/usr/bin/env bash
bash ../install.sh
sbatch basic_job.sh ./non_memory_injection_recovery.py non_memory_res_2.txt
sbatch basic_job.sh ./memory_injection_recovery.py memory_res_2.txt