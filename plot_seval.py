import os
import numpy as np
from secml.adv.seceval import CSecEvalData, CSecEval
from secml.array import CArray
from secml.figure import CFigure
from secml.ml.peval.metrics import CMetricAccuracyReject, CMetricAccuracy

# PARAMETERS
DSET = 'cifar10'
EVAL_TYPE = 'wb'

# ------------------------------------------------------
# CLFS = ['dnn', 'nr', 'dnr', 'tsne_rej', 'tnr']
# CLFS = ['dnn', 'nr']
# CLFS = ['dnn', 'nr', 'dnr_mean']
# CLFS = ['dnn', 'nr', 'rbf_net_sigma_0.000_250', 'rbf_net_nr_like', 'rbf_net_last3']
# CLFS = ['dnn', 'nr', 'dnr'] #, 'rbf_net_sigma_0.000_250', 'rbf_net_fixed_betas', 'rbf_net_sigma_0.100_50', 'rbf_net_no_last']
# CLFS = ['dnn', 'nr', 'dnr', 'rbf_net_sigma_0.000_250', 'deep_rbf_net_sigma_0.000_250']
# CLFS = ['dnn', 'dnr', 'dnr_rbf'] # 'hybrid_rbfnet_svm', 'hybrid_svm_rbfnet',
# CLFS = ['dnn', 'nr', 'dnr', 'rbf_net_sigma_0.000_250', 'deep_rbf_net_sigma_0.000_250', 'dnr_rbf']
# CLFS = ['dnn',
#         'nr',
#         'rbfnet_nr_like_wd_0e+00',w
#         # 'rbfnet_nr_like_wd_1e-03',
#         # 'rbfnet_nr_like_wd_1e-04',
#         # 'rbfnet_nr_like_wd_1e-05',
#         # 'rbfnet_nr_like_wd_1e-06',
#         'rbfnet_nr_like_wd_1e-08',
#         # 'rbfnet_nr_like_wd_1e-10'
#         ]

# if DSET == 'mnist':
#     # MNIST Final
#     CLFS = [
#         'dnn',
#         'nr',
#         'rbfnet_nr_like_10_wd_0e+00',
#         'dnr_asotgiu',
#         'dnr_rbf_tr_init',
#         # 'dnr_rbf_2x',
#     ]
# elif DSET == 'cifar10':
#     # CIFAR10 Final
#     CLFS = [
#         'dnn',
#         'nr',
#         'rbf_net_nr_sv_100_wd_0e+00_cat_hinge_tr_init',
#         'dnr',
#         'dnr_rbf_tr_init',
#         ]
# else:
#     raise ValueError("Unrecognized dataset!")

# CLFS = ['adv_reg_dnn_sigma_0e+00', 'adv_reg_dnn_sigma_1e+03']
# if DSET == 'mnist':
#     CLFS = ['dnn',
#             'nr', 'dnr',
#             'rbfnet_nr_like_10_wd_0e+00',
#             'shallow_fader']
# else:
#     CLFS = ['dnn',
#             'nr', 'dnr',
#             'rbf_net_nr_sv_100_wd_0e+00_cat_hinge_tr_init',
#             'shallow_fader']

# CLFS = ['dnn', 'nr', 'dnr_asotgiu']
# CLFS = ['dnn', 'nr', 'tsne_rej']
# CLFS = ['dnn', 'dnr', 'tnr']
CLFS = ['dnn', 'nr', 'dnr', 'tsne_rej', 'tnr']

if DSET == 'mnist':
    EPS = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    LOG_SCALE = False
elif DSET == 'cifar10':
    EPS = [0.0, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0]
    LOG_SCALE = True
else:
    raise ValueError('Unknown dataset!')

N_ITER = 2     # TODO: RESTORE THIS!

# FNAME = 'all_'+EVAL_TYPE+'_seval'
# FNAME = 'svm_vs_rbf_nr_like'
# FNAME = EVAL_TYPE
# FNAME = 'dnr_rbf_test'
# FNAME = 'adv_reg_dnn_test'
# FNAME = 'fader'
# FNAME = 'buttando_giu_dnr'
# FNAME = 'tsne_rej_'+EVAL_TYPE+'_seval'
# FNAME = 'tnr_'+EVAL_TYPE+'_seval'
FNAME = 'esann19_'+EVAL_TYPE+'_seval'

# DSET = os.path.join(DSET, 'ablation_study')
# EXTENSION = 'png'
EXTENSION = 'pdf'

# COLORS = {
#     'DNN':'tab:blue',
#     'NR':'tab:orange',
#     'NR-RBF': 'tab:green',
#     'DNR': 'tab:red',
#     'DNR-RBF': 'tab:purple',
# }

# COLORS = {
#     'ADV_REG_DNN_SIGMA_0E+00': 'tab:blue',
#     'ADV_REG_DNN_SIGMA_1E+03': 'tab:orange'
# }

# COLORS = {
#     'DNN':'tab:blue',
#     'NR': 'tab:orange',
#     'DNR': 'tab:green',
#     'NR-RBF': 'tab:cyan',
#     'SHALLOW_FADER': 'tab:red'
# }

# COLORS = {
#     'DNN':'tab:blue',
#     'NR': 'tab:orange',
#     'DNR': 'tab:green'
# }

# COLORS = {
#     'DNN':'tab:blue',
#     'NR': 'tab:orange',
#     '$t$-NR': 'tab:green'
# }

# COLORS = {
#     'DNN':'tab:blue',
#     'DNR': 'tab:orange',
#     'D$t$-NR': 'tab:green'
# }

COLORS = {
    'DNN':'tab:blue',
    'NR': 'tab:orange',
    'DNR': 'tab:green',
    '$t$-NR': 'tab:red',
    'D$t$-NR': 'tab:purple'
}

# ------------------------------------------------------


# Plot sec_eval with reject percentage
def rej_percentage(sec_eval_data):
    perf = CArray.zeros(shape=(sec_eval_data.param_values.size,))
    for k in range(sec_eval_data.param_values.size):
        perf[k] = (sec_eval_data.Y_pred[k] == -1).sum() / sec_eval_data.Y_pred[k].shape[0]
    return perf


def acc_rej_performance(sec_eval_data):
    # Compute performance for eps > 0 (i.e. CMetricAccuracyReject)
    acc_metric = CMetricAccuracy()
    rej_metric = CMetricAccuracyReject()

    perf = CArray.zeros(shape=(sec_eval_data.param_values.size,))
    for k in range(sec_eval_data.param_values.size):
        pred = sec_eval_data.Y_pred[k]
        if k == 0:
            # CMetricAccuracy
            metric = acc_metric
        else:
            # CMetricAccuracyReject
            metric = rej_metric
        perf[k] = metric.performance_score(y_true=sec_eval_data.Y, y_pred=pred)
        # print(perf[k])

    return perf


def compute_performance(sec_eval_data, perf_eval_fun):

    if not isinstance(sec_eval_data, list):
        sec_eval_data = [sec_eval_data]

    n_sec_eval = len(sec_eval_data)
    n_param_val = sec_eval_data[0].param_values.size
    perf = CArray.zeros((n_sec_eval, n_param_val))

    # Single runs curves
    for i in range(n_sec_eval):
        if sec_eval_data[i].param_values.size != n_param_val:
            raise ValueError("the number of sec eval parameters changed!")

        perf[i, :] = perf_eval_fun(sec_eval_data[i])

    # Compute mean and std
    perf_std = perf.std(axis=0, keepdims=False)
    perf_mean = perf.mean(axis=0, keepdims=False)
    # perf = perf.ravel()

    return sec_eval_data[i].param_values, perf_mean, perf_std


if __name__ == '__main__':

    fig = CFigure(height=8, width=10)
    # Sec eval plot code
    sp1 = fig.subplot(2, 1, 1)
    sp2 = fig.subplot(2, 1, 2)

    # ============== CLASSIFIERS LOOP ==============
    for clf in CLFS:

        if EVAL_TYPE == 'bb' and clf == 'dnn':
            continue

        # ============== ITERATIONS LOOP ==============
        sec_evals_data = []
        for it in range(N_ITER):
            # Load sec_eval
            fname = os.path.join(DSET, clf + '_'+EVAL_TYPE+'_seval_it_'+str(it)+'.gz')
            if os.path.exists(fname):
                if EVAL_TYPE == 'bb':
                    seval_data = CSecEvalData.load(fname)
                else:
                    seval_data = CSecEval.load(fname).sec_eval_data

                sec_evals_data.append(seval_data)

        # HACK: Changing plot classifier names
        if clf == 'tsne_rej':
            label = '$t$-NR'
        elif clf == 'tnr':
            label = 'D$t$-NR'
        # DEBUG: REMOVE THIS!
        # DEBUG: ========================
        # elif clf == 'nr':
        #     label = 'svm-rbf'
        # elif clf == 'rbfnet_nr_like':
        #     label = 'rbfnet'
        # elif clf == 'rbfnet_nr_like_wd_0e+00':
        #     label = 'rbfnet_nr_like_no_reg'
        elif 'rbfnet' in clf or 'rbf_net' in clf:
            label = 'NR-RBF'
        elif clf == 'dnr_rbf_tr_init':
            label = 'DNR-RBF'
        elif clf == 'dnr_asotgiu':
            label = 'DNR'
        # DEBUG: ========================
        else:
            label = clf.upper()

        print(" - Plotting ", label)

        # Plot performance
        _, perf, perf_std = compute_performance(sec_evals_data, acc_rej_performance)
        sp1.plot(EPS, perf, label=label, color=COLORS[label], marker="o")
        # Plot mean and std
        std_up = perf + perf_std
        std_down = perf - perf_std
        std_down[std_down < 0.0] = 0.0
        std_down[std_up > 1.0] = 1.0
        sp1.fill_between(EPS, std_up, std_down, interpolate=False, alpha=0.2)

        # # Convenience function for plotting the Security Evaluation Curve
        # sp1.plot_sec_eval(sec_evals_data, marker='o', label=label, mean=True, #show_average=True,
        #                   metric=['accuracy'] + ['accuracy-reject'] * (sec_evals_data[0].param_values.size - 1))


        # Plot reject percentage
        _, perf, perf_std = compute_performance(sec_evals_data, rej_percentage)
        sp2.plot(EPS, perf, label=label, color=COLORS[label], marker="o")
        # Plot mean and std
        std_up = perf + perf_std
        std_down = perf - perf_std
        std_down[std_down < 0.0] = 0.0
        std_down[std_up > 1.0] = 1.0
        sp2.fill_between(EPS, std_up, std_down, interpolate=False, alpha=0.2)

    if LOG_SCALE:
        sp1.xscale('symlog', linthreshx=0.1)
    sp1.xticks(EPS)
    sp1.xticklabels(EPS)
    sp1.ylim(-0.05, 1.05)
    sp1.xlabel('dmax')
    sp1.ylabel('Accuracy') #(CMetricAccuracyReject().class_type.capitalize())
    sp1.legend()
    sp1.title("{0} evasion attack ({1})".format(
        'Black-box' if EVAL_TYPE == 'bb' else 'White-box',
        DSET.split('/')[0].upper()  # HACK: Ablation study needed this!
    ))
    sp1.apply_params_sec_eval()

    if LOG_SCALE:
        sp2.xscale('symlog', linthreshx=0.1)
    sp2.xticks(EPS)
    sp2.xticklabels(EPS)
    # sp2.ylim(-0.05, 1.05)
    sp2.xlabel('dmax')
    sp2.ylabel('Rejection rate') #("% Reject")
    sp2.apply_params_sec_eval()

    # Dump to file
    out_file = os.path.join(DSET, FNAME+'.'+EXTENSION)
    print("- Saving output to file: {}".format(out_file))
    fig.savefig(out_file, file_format=EXTENSION)
