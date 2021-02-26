import h5py
import numpy as np

from memestr.events import events


events_official = [
    "GW190408_181802.h5", "GW190412.h5",  "GW190413_052954.h5", "GW190413_134308.h5", "GW190421_213856.h5",
    "GW190424_180648.h5", #"GW190425.h5", "GW190426_152155.h5",
    "GW190503_185404.h5", "GW190512_180714.h5",
    "GW190513_205428.h5", "GW190514_065416.h5", "GW190517_055101.h5", "GW190519_153544.h5", "GW190521.h5",
    "GW190521_074359.h5", "GW190527_092055.h5", "GW190602_175927.h5", "GW190620_030421.h5", "GW190630_185205.h5",
    "GW190701_203306.h5", "GW190706_222641.h5", "GW190707_093326.h5", "GW190708_232457.h5", #"GW190719_215514.h5",
    "GW190720_000836.h5", "GW190727_060333.h5", "GW190728_064510.h5", "GW190731_140936.h5", "GW190803_022701.h5",
    "GW190814.h5", "GW190828_063405.h5", "GW190828_065509.h5", #"GW190909_114149.h5",
    "GW190910_112807.h5", "GW190915_235702.h5", "GW190924_021846.h5", "GW190929_012149.h5", "GW190930_133541.h5"]

for i in range(len(events_official)):
    event_name_short = events[i + 10].name
    event_name_official = events_official[i]
    print(event_name_short[:8])
    print(event_name_official[:8])
    assert event_name_short[:8] == event_name_official[:8]

    filename = f"all_posterior_samples/{event_name_official}"
    with h5py.File(filename, "r") as f:
        # List all groups
        # print("Keys: %s" % f.keys())
        # print()
        for j in range(len(list(f.keys()))):
            a_group_key = list(f.keys())[j]
            print(a_group_key)
            # print()

            # Get the data
            data = list(f[a_group_key])
            print(f[a_group_key])
            print()
            try:
                print(f[a_group_key]['psds'])
                break
            except Exception:
                print(f"{f[a_group_key]} has no psd")
                continue
        if 'H1' in events[i].detectors:
            try:
                H1_freqs = f[a_group_key]['psds']['H1'][:, 0]
                H1_powers = f[a_group_key]['psds']['H1'][:, 1]
                H1_file = np.array([H1_freqs, H1_powers]).T
                np.savetxt(f"GWTC2_PSDs/{event_name_short}_LIGO_Hanford_psd.txt", H1_file)
            except KeyError:
                pass
        if 'L1' in events[i].detectors:
            try:
                L1_freqs = f[a_group_key]['psds']['L1'][:, 0]
                L1_powers = f[a_group_key]['psds']['L1'][:, 1]
                L1_file = np.array([L1_freqs, L1_powers]).T
                np.savetxt(f"GWTC2_PSDs/{event_name_short}_LIGO_Livingston_psd.txt", L1_file)
            except KeyError:
                pass
        if 'V1' in events[i].detectors:
            try:
                V1_freqs = f[a_group_key]['psds']['V1'][:, 0]
                V1_powers = f[a_group_key]['psds']['V1'][:, 1]
                V1_file = np.array([V1_freqs, V1_powers]).T
                np.savetxt(f"GWTC2_PSDs/{event_name_short}_Virgo_psd.txt", V1_file)
            except KeyError as e:
                pass

