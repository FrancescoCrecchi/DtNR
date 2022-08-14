from secml.array import CArray
from secml.ml import CClassifierSVM, CKernelRBF, CNormalizerMinMax
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.classifiers.reject import CClassifierDNR

from components.c_classifier_kde import CClassifierKDE
from components.c_reducer_ptsne import CReducerPTSNE

from mnist.cnn_mnist import cnn_mnist_model
from mnist.fit_dnn import get_datasets

# LOGFILE = 'tnr_best_params.log'


N_TRAIN, N_TEST = 10000, 1000
if __name__ == '__main__':
    random_state = 999

    _, vl, ts = get_datasets(random_state)

    # Load classifier
    dnn = cnn_mnist_model()
    dnn.load_model('cnn_mnist.pkl')

    # Check test performance
    y_pred = dnn.predict(ts.X, return_decision_function=False)

    from secml.ml.peval.metrics import CMetric
    acc_torch = CMetric.create('accuracy').performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc_torch))

    # Create LD
    tsne = CReducerPTSNE(n_hiddens=[128, 128], epochs=500, batch_size=32, preprocess=None, random_state=random_state)
    LD = CClassifierMulticlassOVA(classifier=CClassifierKDE, kernel=CKernelRBF(), preprocess=tsne)

    # Create DNR
    layers = ['features:relu2', 'features:relu3', 'features:relu4']
    combiner = CClassifierMulticlassOVA(CClassifierSVM, kernel=CKernelRBF())
    layer_clf = LD
    tnr = CClassifierDNR(combiner, layer_clf, dnn, layers, -1000)

    # Setting layer classifiers parameters (separate xval)
    tnr.set_params({
        'features:relu2.kernel.gamma': 0.1,
        'features:relu3.kernel.gamma': 0.1,
        'features:relu4.kernel.gamma': 0.1,
        'clf.C': 10,
        'clf.kernel.gamma': 0.01
    })

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # DEBUG: CClassifierPTSNE Verbose
    # [lcf.preprocess.__class__ for lcf in tnr._layer_clfs.values()]
    # for lcf in tnr._layer_clfs.values():
    #     lcf.preprocess.verbose = 1

    # # Parallelize?
    # tnr.n_jobs = 4

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
