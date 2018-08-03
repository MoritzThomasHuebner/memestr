#!/usr/bin/env bash
bash ../install.sh
sbatch non_memory_injection.sh ./non_memory_injection_recovery.py non_memory_res_2.txt
sbatch memory_injection.sh ./memory_injection_recovery.py memory_res_2.txt