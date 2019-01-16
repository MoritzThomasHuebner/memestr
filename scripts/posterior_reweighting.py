import bilby as bb
import memestr

result = bb.result.read_in_result(outdir='posterior_reweighting', label='IMR_mem_inj_mem_rec')

print(result)