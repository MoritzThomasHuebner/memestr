import bilby

from memestr.events import events, precessing_events


# for e in events:
#     try:
#         time_tag = e.time_tag
#         event = e.name
#         detectors = e.detectors
#         result = bilby.core.result.read_in_result(f'{event}/result/run_data0_{time_tag}_analysis_{detectors}_dynesty_merge_result.json')
#         result.outdir = f'.'
#         result.label = f"{e.name}_phenomx"
#         result.plot_corner(filename=f'corner_plots/{e.name}_phenomx.png')
#     except Exception as ex:
#         print(ex)

for e in precessing_events:
    try:
        time_tag = e.time_tag
        event = e.name
        detectors = e.detectors
        result = bilby.core.result.read_in_result(f'{event}/result/run_data0_{time_tag}_analysis_{detectors}_dynesty_merge_result.json')
        result.plot_corner(filename=f'corner_plots/{e.name}_nrsur.png')
    except Exception as e:
        print(e)
