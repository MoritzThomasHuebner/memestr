from memestr.events import events, precessing_events


def get_detector_list(detectors):
    return [event.detectors[2*i: 2*i+2] for i in range(int(len(event.detectors)/2))]


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



for event in events:
    trigger_time = event.time_tag.replace('-', '.')

    detectors_list = get_detector_list(event.detectors)
    detector_string = get_detector_string(detectors_list)

    channel_dict_string = repr({det: 'GWOSC' for det in detectors_list}).replace(": ", ":").replace("\'", "")

    psd_dict_string = get_psd_dict_string(detectors_list, event.name)
    # assert False
    with open(f"{event.name}.ini", 'w') as f:
        f.write(f"trigger-time = {trigger_time}\n")
        f.write(f"outdir = {event.name}\n")
        f.write(f"detectors = {detector_string}\n")
        f.write(f"duration = {event.duration}\n")
        f.write(f"channel-dict = {channel_dict_string}\n")
        f.write(f"psd-dict = {psd_dict_string}\n")
        f.write("\n")
        f.write("\n")
        f.write("label = run\n")
        f.write("accounting = ligo.dev.o3.cbc.pe.lalinference\n")
        f.write("scheduler = slurm\n")
        f.write("scheduler-args = mem=4G\n")
        f.write("scheduler-module = python/3.7.4\n")
        f.write("scheduler-analysis-time = 5-00:00:00\n")
        f.write("submit=True\n")
        f.write("\n")
        f.write("\n")
        f.write("coherence-test = False\n")
        f.write("\n")
        f.write("sampling-frequency=2048\n")
        f.write("frequency-domain-source-model = memestr.waveforms.phenom.fd_imrx_fast\n")
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

for event in precessing_events:
    trigger_time = event.time_tag.replace('-', '.')

    detectors_list = get_detector_list(event.detectors)
    detector_string = get_detector_string(detectors_list)

    channel_dict_string = repr({det: 'GWOSC' for det in detectors_list}).replace(": ", ":").replace("\'", "")

    psd_dict_string = get_psd_dict_string(detectors_list, event.name)

    with open(f"{event.name}.ini", 'w') as f:
        f.write(f"trigger-time = {trigger_time}\n")
        f.write(f"outdir = {event.name}_2000\n")
        f.write(f"detectors = {detector_string}\n")
        f.write(f"duration = {event.duration}\n")
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
        f.write("frequency-domain-source-model = memestr.waveforms.nrsur7dq4.fd_nr_sur_7dq4\n")
        f.write("\n")
        f.write("\n")
        f.write("calibration-model=None\n")
        f.write("\n")
        f.write("minimum-frequency=0\n")
        f.write("reference-frequency=50\n")
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