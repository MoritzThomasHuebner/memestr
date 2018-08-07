#!/usr/bin/env bash
bash ../install.sh
sbatch non_memory_injection.sh
sbatch memory_injection.sh