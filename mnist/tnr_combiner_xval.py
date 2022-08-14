from secml.array import CArray
from secml.data import CDataset
from secml.ml import CKernelRBF, CNormalizerMinMax, CClassifierSVM
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.classifiers.reject import CClassifierDNR
from secml.ml.peval import CPerfEvaluatorXVal

from components.c_classifier_kde import CClassifierKDE
from components.c_reducer_ptsne import CReducerPTSNE
from mnist.cnn_mnist import cnn_mnist_model
from mnist.fit_dnn import get_datasets

N_TRAIN, N_TEST = 10000, 1000
LOGFILE = 'tnr_best_params.log'
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

    # # Create LD
    # tsne = CReducerPTSNE(n_hiddens=[128, 128], epochs= 500, batch_size=32, preprocess=None, random_state=random_state)
    # LD = CClassifierMulticlassOVA(classifier=CClassifierKDE, kernel=CKernelRBF(), preprocess=tsne, n_jobs=8)
    #
    # # Create DNR
    # layers = ['features:relu2', 'features:relu3', 'features:relu4']
    combiner = CClassifierMulticlassOVA(CClassifierSVM, kernel=CKernelRBF())
    # layer_clf = LD
    # tnr = CClassifierDNR(combiner, layer_clf, dnn, layers, -1000)
    #
    # # Setting layer classifiers parameters (separate xval)
    # tnr.set_params({
    #     'features:relu2.kernel.gamma': 0.1,
    #     'features:relu3.kernel.gamma': 0.1,
    #     'features:relu4.kernel.gamma': 0.1
    # })

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # Load TNR
    tnr = CClassifierDNR.load('tnr.gz')

    # Parallel?
    tnr.n_jobs = 4

    # Obtain intermediate representations
    comb_X = tnr._create_scores_dataset(tr_sample.X, tr_sample.Y)
    comb_dset = CDataset(comb_X, tr_sample.Y)

    # Xval
    xval_params = {'C': [1e-2, 1e-1, 1, 10, 100],
                   'kernel.gamma': [1e-2, 1e-1, 1]}

    # Let's create a 3-Fold data splitter
    from secml.data.splitter import CDataSplitterKFold
    xval_splitter = CDataSplitterKFold(num_folds=3, random_state=random_state)

    # # Now we compare the parameters chosen before with a new evaluator
    # perf_eval = CPerfEvaluatorXVal(xval_splitter, CMetric.create('accuracy'))
    # perf_eval.verbose = 1

    # Parallelize?
    # combiner.n_jobs = 1

    # Select and set the best training parameters for the classifier
    print("Estimating the best training parameters...")
    # combiner.verbose = 1
    best_params = combiner.estimate_parameters(
        dataset=comb_dset,
        parameters=xval_params,
        splitter=xval_splitter,
        metric='accuracy',
        verbose = True
    )

    print("The best training parameters are: ",
          [(k, best_params[k]) for k in sorted(best_params)])

    # Dump to disk
    with open(LOGFILE, "a") as f:
        f.write("COMBINER best params: {:} \n".format([(k, best_params[k]) for k in sorted(best_params)]))
