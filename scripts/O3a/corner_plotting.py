import bilby

from memestr.events import events


for e in events:
    try:
        time_tag = e.time_tag
        event = e.name
        detectors = e.detectors
        result = bilby.core.result.read_in_result(f'{event}/result/run_data0_{time_tag}_analysis_{detectors}_dynesty_merge_result.json')
        result.outdir = f'.'
        result.plot_corner()
    except Exception as ex:
        print(ex)