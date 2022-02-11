import bilby

from memestr.events import events, precessing_events


for e in events:
    if e.name not in ['GW191126A', 'GW191216A', 'GW200202A', 'GW200210A', 'GW200220A']:
        continue
    try:
        time_tag = e.time_tag
        event = e.name
        detectors = e.detectors
        result = bilby.core.result.read_in_result(f'{event}/result/run_data0_{time_tag}_analysis_{detectors}_dynesty_merge_result.json')
        result.outdir = f'.'
        result.label = f"{e.name}_phenomx"
        result.plot_corner(filename=f'corner_plots/{e.name}_phenomx.png')
    except Exception as ex:
        print(ex)

for e in precessing_events:
    if e.name not in ['GW191105A_prec', 'GW191126A_prec', 'GW200202A_prec', 'GW200220A_prec']:
        continue
    try:
        time_tag = e.time_tag
        event = e.name
        detectors = e.detectors
        result = bilby.core.result.read_in_result(f'{event}_2000/result/run_data0_{time_tag}_analysis_{detectors}_dynesty_merge_result.json')
        result.plot_corner(filename=f'corner_plots/{e.name}_nrsur.png')
    except Exception as e:
        print(e)
