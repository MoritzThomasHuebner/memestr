import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import bilby
import corner

matplotlib.rcParams.update(matplotlib.rcParamsDefault)

# plt.style.use('paper.mplstyle')
event = "GW191109A"

log_weights_aligned = np.loadtxt(f"{event}_memory_log_weights")
# log_weights_prec = np.loadtxt(f"data/{event}_prec_2000_memory_log_weights")


res_aligned = bilby.result.read_in_result(f"{event}/result/run_data0_1260567236-4_analysis_H1V1_dynesty_merge_result.json")
# res_aligned = bilby.result.read_in_result(f"data/{event}_result.json")
# res_prec = bilby.result.read_in_result(f"data/{event}_prec_result.json")
# res_prec_reweighted = bilby.result.read_in_result(f"data/{event}_prec_result.json")
# print(len(res_prec_reweighted.posterior))
# print(len(log_weights_prec))


res_aligned.posterior['memory_log_weight'] = log_weights_aligned
# res_prec.posterior['memory_log_weight'] = log_weights_prec

defaults_kwargs = dict(
    bins=50, smooth=0.9, label_kwargs=dict(fontsize=16),
    title_kwargs=dict(fontsize=16), color='#0072C1',
    truth_color='tab:orange', quantiles=[0.16, 0.84],
    levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    plot_density=False, plot_datapoints=True, fill_contours=True,
    max_n_ticks=3, hist_kwargs=dict(density=True))


params = ['total_mass', 'mass_ratio', 'inc', 'luminosity_distance', 'phase', 'psi', 'ra', 'dec', 'geocent_time', 'a_1',
          'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 's13', 's23']
labels = ['total mass', 'mass ratio', r'$\theta_{\mathrm{JN}}\, \mathrm{[rad]}$', r'$d_L\, \mathrm{[Mpc]}$', 'phase', 'psi',
          'ra', 'dec', 'geocent time', 'a 1', 'a 2', 'tilt 1', 'tilt 2', 'phi 12', 'phi jl', 's13', 's23']


matplotlib.rc('font', **{'family': 'serif', 'serif': ['Times']})
matplotlib.rc('text', usetex=True)
for param, param_label in zip(params, labels):
    plt.figure(dpi=150)
    plt.xlabel(param_label)
    plt.ylabel(r'$\ln w_{\mathrm{mem}}$')
    try:
        corner.hist2d(np.array(res_aligned.posterior[param]), log_weights_aligned,
                      labels=[r"memory log weights", "param"],
                      bins=50, smooth=0.9, label_kwargs=dict(fontsize=16), title_kwargs=dict(fontsize=16), color='C0',
                      truth_color='tab:orange', quantiles=[0.16, 0.84],
                      levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                      plot_density=False, plot_datapoints=True, fill_contours=True, max_n_ticks=3,
                      hist_kwargs=dict(density=True))
    except Exception:
        pass
    # try:
    #     corner.hist2d(np.array(res_prec.posterior[param]), log_weights_prec, labels=[r"memory log weights", "param"],
    #                   bins=50, smooth=0.9, label_kwargs=dict(fontsize=16), title_kwargs=dict(fontsize=16), color='C1',
    #                   truth_color='tab:orange', quantiles=[0.16, 0.84],
    #                   levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
    #                   plot_density=False, plot_datapoints=True, fill_contours=True, max_n_ticks=3,
    #                   hist_kwargs=dict(density=True))
    # except Exception:
    #     pass
    lgnd = plt.legend(labels=['IMRPhenomXHM', 'NRSur7dq4'], markerscale=5)
    if param == 'inc':
        plt.xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi], (r'$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'))
    plt.tight_layout()
    plt.savefig(f'corner_plots/{event}_{param}.png')
    plt.clf()



