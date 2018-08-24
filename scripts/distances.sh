#!/usr/bin/env bash
bash ../install.sh
for i in `seq 50 100 500`;
        do
                bash IMR_mem_inj_mem_rec.sh ${i}
                bash IMR_mem_inj_non_mem_rec.sh ${i}
                bash IMR_non_mem_inj_non_mem_rec.sh ${i}
                bash IMR_non_mem_inj_mem_rec.sh ${i}
                bash IMR_pure_mem_inj_pure_mem_rec.sh ${i}
        done