from secml.array import CArray
from secml.data import CDataset
from secml.ml import CKernelRBF, CNormalizerMinMax, CClassifierSVM, CNormalizerMeanStd
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.classifiers.reject import CClassifierDNR
from secml.ml.peval.metrics import CMetricAccuracy

from components.c_classifier_kde import CClassifierKDE
from components.c_reducer_ptsne import CReducerPTSNE

from cifar10.cnn_cifar10 import cifar10
from cifar10.fit_dnn import get_datasets

N_TRAIN, N_TEST = 10000, 1000
LOGFILE = 'tnr_best_params.log'
if __name__ == '__main__':
    random_state = 999

    _, vl, ts = get_datasets(random_state)

    # Load classifier
    dnn = cifar10(preprocess=CNormalizerMeanStd(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    dnn.load_model('cnn_cifar10.pkl')

    # Check test performance
    y_pred = dnn.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc))

    # Create LD
    tsne = CReducerPTSNE(epochs=500, batch_size=32, preprocess=None, random_state=random_state)
    tsne.verbose = 1
    LD = CClassifierMulticlassOVA(classifier=CClassifierKDE, kernel=CKernelRBF(), preprocess=tsne, n_jobs=10)

    # Create DNR
    layers = ['features:26', 'features:29'] # 'features:23',
    combiner = CClassifierMulticlassOVA(CClassifierSVM, kernel=CKernelRBF())
    layer_clf = LD

    tnr = CClassifierDNR(combiner, layer_clf, dnn, layers, -1000)
    tnr.verbose = 1

    # Setting layer classifiers parameters (separate xval)
    tnr.set_params({
        # 'features:23.preprocess.preprocess.n_hiddens': [256],
        # 'features:23.kernel.gamma': 100,
        'features:26.preprocess.n_hiddens': [256, 256],
        'features:26.kernel.gamma': 0.1,
        'features:29.preprocess.n_hiddens': [256, 256],
        'features:29.kernel.gamma': 0.1,
    })

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # Parallelize?
    # tnr.n_jobs = 10

    # Obtain intermediate representations
    print("[DEBUG] Creating 'comb_dset'...")
    comb_X = tnr._create_scores_dataset(tr_sample.X, tr_sample.Y)
    comb_dset = CDataset(comb_X, tr_sample.Y)
    comb_dset.save('comb_dset')     # DEBUG: Dump to disk
    print("[DEBUG] comb_dset dumped to disk.")

    # Checkpoint save
    # Dump to disk
    tnr.save('tnr_PRE_COMB_XVAL')

    # # LOAD
    # tnr = CClassifierDNR.load('tnr_PRE_COMB_XVAL.gz')
    # combiner = tnr.clf
    # assert not combiner.is_fitted(), "Something wrong here!"

    # Xval
    xval_params = {'C': [1e-2, 1e-1, 1, 10, 100],
                   'kernel.gamma': [1e-3, 1e-2, 1e-1, 1]}

    # Let's create a 3-Fold data splitter
    from secml.data.splitter import CDataSplitterKFold
    xval_splitter = CDataSplitterKFold(num_folds=3, random_state=random_state)

    # Select and set the best training parameters for the classifier
    print("Estimating the best training parameters...")
    best_params = combiner.estimate_parameters(
        dataset=comb_dset,
        parameters=xval_params,
        splitter=xval_splitter,
        metric='accuracy',
        verbose=1
    )

    print("The best training parameters are: ",
          [(k, best_params[k]) for k in sorted(best_params)])

    # Dump to disk
    with open(LOGFILE, "a") as f:
        f.write("COMBINER best params: {:} \n".format([(k, best_params[k]) for k in sorted(best_params)]))

    # Fit DNR
    tnr.fit(tr_sample.X, tr_sample.Y)
    # Set threshold (FPR: 10%)
    tnr.threshold = tnr.compute_threshold(0.1, ts_sample)

    # Check test performance
    y_pred = tnr.predict(ts.X, return_decision_function=False)

    from secml.ml.peval.metrics import CMetric
    acc_torch = CMetric.create('accuracy').performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc_torch))

    # Dump to disk
    tnr.save('tnr')