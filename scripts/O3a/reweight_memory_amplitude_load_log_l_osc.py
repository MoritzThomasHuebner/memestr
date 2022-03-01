import bilby.core.result
import pandas as pd
import memestr

for event in memestr.events.events:
    csv_file = f'memory_amplitude_results/{event.name}_memory_amplitude_posterior_50.csv'
    posterior = pd.read_csv(csv_file)
    result = bilby.core.result.read_in_result(
        f'{event.name}/result/run_data0_{event.time_tag}_analysis_{event.detectors}_dynesty_merge_result.json')
    posterior['log_l_osc'] = result.posterior['log_likelihood']
    posterior.to_csv(csv_file, index=False)
    print(event.name)
