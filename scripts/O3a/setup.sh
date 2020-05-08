for EVENT in GW150914 GW151012 GW151226 GW170104 GW170608 GW170729 GW170809 GW170814 GW170818 GW170823
do
  bilby_pipe ${EVENT}.ini
  bash ${EVENT}/submit/run_data0_*_generation.sh
  FILENAME=$(find "./${EVENT}/submit/" -iname "*generation.sh")
  echo '#!/bin/bash' >  "${FILENAME}"
  echo 'echo 0' >> "${FILENAME}"
  sbatch ${EVENT}/submit/run_master_slurm.sh
done
