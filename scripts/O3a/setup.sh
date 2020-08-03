for EVENT in GW150914 GW151012 GW151226 GW170104 GW170608 GW170729 GW170809 GW170814 GW170818 GW170823 GW190412
do
  bilby_pipe ${EVENT}_22.ini
  bash ${EVENT}_precessing_22/submit/run_data0_*_generation.sh
  FILENAME=$(find "./${EVENT}_precessing_22/submit/" -iname "*generation.sh")
  echo '#!/bin/bash' >  "${FILENAME}"
  echo 'echo 0' >> "${FILENAME}"
  echo 'periodic-restart-time = 1209600' >> "${EVENT}_precessing_22/run_config_complete.ini"
  sbatch ${EVENT}_precessing_22/submit/run_master_slurm.sh
done
