import bilby
import memestr
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib
import pickle


sampling_frequency = 2048
duration = 4
series = bilby.core.series.CoupledTimeAndFrequencySeries(sampling_frequency=sampling_frequency, duration=duration)

times = series.time_array
frequencies = series.frequency_array


with open(f'data/GW190521A_prec/data/run_data0_1242459857-5_generation_data_dump.pickle', "rb") as f:
    data_dump = pickle.load(f)
ifos = data_dump.interferometers

res_prec = bilby.result.read_in_result("data/GW190521A_prec/result/run_data0_1242459857-5_analysis_H1L1_dynesty_merge_result.json")
res_aligned = bilby.result.read_in_result("data/GW190521A/result/run_data0_1242459857-5_analysis_H1L1_dynesty_merge_result.json")

wg_xhm = bilby.gw.waveform_generator.WaveformGenerator(
    sampling_frequency=ifos.sampling_frequency, duration=ifos.duration,
    frequency_domain_source_model=memestr.waveforms.phenom.xhm.fd_imrx)

wg_nrsur = bilby.gw.waveform_generator.WaveformGenerator(
    sampling_frequency=ifos.sampling_frequency, duration=ifos.duration,
    frequency_domain_source_model=memestr.waveforms.fd_nr_sur_7dq4)


from scipy.optimize import minimize

def calculate_overlaps_optimizable(new_params, *args):
    phase = new_params[0]
    wg_xhm, wg_nrsur, frequency_array, power_spectral_density = args
    wg_nrsur.parameters['phase'] = phase

    xhm = wg_xhm.frequency_domain_strain()
    nrsur = wg_nrsur.frequency_domain_strain()

    return -memestr.postprocessing.overlap_function(a=xhm, b=nrsur, frequency=frequency_array, psd=power_spectral_density)

res_aligned_shifted = deepcopy(res_aligned)

for i in range(len(res_aligned.posterior)):
    new_params = res_aligned.posterior.iloc[i].to_dict()
    new_params["a_1"] = new_params["s13"]
    new_params["a_2"] = new_params["s23"]
    new_params["tilt_1"] = 0.0
    new_params["tilt_2"] = 0.0
    new_params["phi_12"] = 0.0
    new_params["phi_jl"] = 0.0

    wg_xhm.parameters = new_params
    wg_nrsur.parameters = new_params
    # max_overlap = -1
    # max_overlap_phase = new_params['phase']

    max_overlap = -1.
    counter = 0
    while max_overlap < 0.95:
        x0 = np.array([np.random.random() * 2*np.pi])
        args = (wg_xhm, wg_nrsur, series.frequency_array, ifos[0].power_spectral_density)
        bounds = [(0, 2*np.pi)]
        res = minimize(calculate_overlaps_optimizable, x0=x0, args=args, bounds=bounds, tol=0.001)
        max_overlap = -res.fun
        max_overlap_phase = res.x[0]
        counter += 1
        if counter > 10:
            break

    # for new_phase in np.arange(0, 2*np.pi, 2*np.pi/33):
    #     new_new_params = new_params.copy()
    #     new_new_params['phase'] = new_phase
    #     wg_nrsur.parameters = new_new_params
    #     overlap = memestr.postprocessing.overlap_function(a=wg_xhm.frequency_domain_strain(), b=wg_nrsur.frequency_domain_strain(), frequency=wg_xhm.frequency_array, psd=ifos[0].power_spectral_density)
    #     if overlap > max_overlap:
    #         max_overlap = overlap
    #         max_overlap_phase = new_phase
    #     if max_overlap > 0.99:
    #         break
    res_aligned_shifted.posterior['phase'].iloc[i] = max_overlap_phase
    print(i/len(res_aligned.posterior))
    print(max_overlap)
    print(np.abs(new_params['phase'] - max_overlap_phase))
    print()

res_aligned_shifted.label = 'res_aligned_shifted'
res_aligned_shifted.outdir = '.'
res_aligned_shifted.save_to_file()
