#!/usr/bin/env bash
bash ../install.sh
sbatch non_memory_injection.sh ./non_memory_injection_recovery.py
sbatch memory_injection.sh ./memory_injection_recovery.py