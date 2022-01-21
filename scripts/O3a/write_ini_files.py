from memestr.events import events, precessing_events


def get_detector_list():
    return [e.detectors[2 * i: 2 * i + 2] for i in range(int(len(e.detectors) / 2))]


def get_detector_string(detectors):
    ret = ""
    for d in detectors:
        ret += f"{d}, "
    return f"[{ret.rstrip(', ')}]"


def get_psd_dict_string(detectors, event):
    event = event.rstrip('_prec')
    psd_dict = {}
    if "H1" in detectors:
        psd_dict["H1"] = f"GWTC2_PSDs/{event}_LIGO_Hanford_psd.txt"
    if "L1" in detectors:
        psd_dict["L1"] = f"GWTC2_PSDs/{event}_LIGO_Livingston_psd.txt"
    if "V1" in detectors:
        psd_dict["V1"] = f"GWTC2_PSDs/{event}_Virgo_psd.txt"
    return repr(psd_dict).replace(": ", ":").replace("\'", "")


for e in events:
    trigger_time = e.time_tag.replace('-', '.')

    detectors_list = get_detector_list()
    detector_string = get_detector_string(detectors_list)

    channel_dict_string = repr({det: 'GWOSC' for det in detectors_list}).replace(": ", ":").replace("\'", "")

    psd_dict_string = get_psd_dict_string(detectors_list, e.name)
    with open(f"{e.name}.ini", 'w') as f:
        f.write(f"trigger-time = {trigger_time}\n")
        f.write(f"outdir = {e.name}_memory_amplitude_2000\n")
        f.write(f"detectors = {detector_string}\n")
        f.write(f"duration = {e.duration}\n")
        f.write(f"channel-dict = {channel_dict_string}\n")
        f.write(f"psd-dict = {psd_dict_string}\n")
        f.write("\n")
        f.write("\n")
        f.write("label = run\n")
        f.write("accounting = ligo.dev.o3.cbc.pe.lalinference\n")
        f.write("scheduler = slurm\n")
        f.write("scheduler-args = mem=4G\n")
        f.write("scheduler-module = python/3.7.4\n")
        f.write("scheduler-analysis-time = 7-00:00:00\n")
        f.write("submit=True\n")
        f.write("\n")
        f.write("\n")
        f.write("coherence-test = False\n")
        f.write("\n")
        f.write("sampling-frequency=2048\n")
        f.write("frequency-domain-source-model = memestr.waveforms.phenom.fd_imrx_with_memory\n")
        f.write("\n")
        f.write("\n")
        f.write("calibration-model=None\n")
        f.write("\n")
        f.write("deltaT = 0.2\n")
        f.write("time-marginalization=False\n")
        f.write("distance-marginalization=False\n")
        f.write("phase-marginalization=False\n")
        f.write("\n")
        f.write("prior-file = aligned_spin.prior\n")
        f.write("\n")
        f.write("sampler = dynesty\n")
        f.write("sampler-kwargs = {nlive: 2000, sample: rwalk, walks=100, n_check_point=2000, nact=10, resume=True}\n")
        f.write("\n")
        f.write("n-parallel = 5\n")
        f.write("\n")
        f.write("create-plots=True\n")
        f.write("local-generation = True\n")
        f.write("transfer-files = False\n")
        f.write("periodic-restart-time = 1209600")

for e in precessing_events:
    if e.name == "GW190521_prec":
        reference_frequency = 11
    else:
        reference_frequency = 20
    trigger_time = e.time_tag.replace('-', '.')

    detectors_list = get_detector_list()
    detector_string = get_detector_string(detectors_list)

    channel_dict_string = repr({det: 'GWOSC' for det in detectors_list}).replace(": ", ":").replace("\'", "")

    psd_dict_string = get_psd_dict_string(detectors_list, e.name)

    with open(f"{e.name}.ini", 'w') as f:
        f.write(f"trigger-time = {trigger_time}\n")
        f.write(f"outdir = {e.name}_memory_amplitude_2000\n")
        f.write(f"detectors = {detector_string}\n")
        f.write(f"duration = {e.duration}\n")
        f.write(f"channel-dict = {channel_dict_string}\n")
        f.write(f"psd-dict = {psd_dict_string}\n")
        f.write("\n")
        f.write("\n")
        f.write("label = run\n")
        f.write("accounting = ligo.dev.o3.cbc.pe.lalinference\n")
        f.write("scheduler = slurm\n")
        f.write("scheduler-args = mem=4G\n")
        f.write("scheduler-module = python/3.7.4\n")
        f.write("scheduler-analysis-time = 7-00:00:00\n")
        f.write("submit=True\n")
        f.write("\n")
        f.write("\n")
        f.write("coherence-test = False\n")
        f.write("\n")
        f.write("sampling-frequency=2048\n")
        f.write("frequency-domain-source-model = memestr.waveforms.nrsur7dq4.fd_nr_sur_7dq4_with_memory\n")
        f.write("\n")
        f.write("\n")
        f.write("calibration-model=None\n")
        f.write("\n")
        f.write("minimum-frequency=0\n")
        f.write(f"reference-frequency={reference_frequency}\n")
        f.write("deltaT = 0.2\n")
        f.write("time-marginalization=False\n")
        f.write("distance-marginalization=False\n")
        f.write("phase-marginalization=False\n")
        f.write("\n")
        f.write("prior-file = precessing_spin.prior\n")
        f.write("\n")
        f.write("sampler = dynesty\n")
        f.write("sampler-kwargs = {nlive: 2000, sample: rwalk, walks=100, n_check_point=2000, nact=10, resume=True}\n")
        f.write("\n")
        f.write("n-parallel = 5\n")
        f.write("\n")
        f.write("create-plots=True\n")
        f.write("local-generation = True\n")
        f.write("transfer-files = False\n")
        f.write("periodic-restart-time = 1209600")
