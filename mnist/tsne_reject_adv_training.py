from secml.array import CArray
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.classifiers.reject import CClassifierRejectThreshold
from secml.ml.features import CNormalizerDNN
from secml.ml.kernels import CKernelRBF
from secml.ml.peval.metrics import CMetricAccuracy

from components.c_classifier_kde import CClassifierKDE
from components.c_reducer_ptsne import CReducerPTSNE
from mnist.cnn_mnist import cnn_mnist_model
from mnist.fit_dnn import get_datasets
from mnist.tsne_rej_inspector import load_adv_ds

N_TRAIN, N_TEST = 10000, 1000
if __name__ == '__main__':
    random_state = 999

    _, vl, ts = get_datasets(random_state)

    # Load classifier
    dnn = cnn_mnist_model()
    dnn.load_model('cnn_mnist.pkl')

    # Check test performance
    y_pred = dnn.predict(ts.X, return_decision_function=False)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc))

    # Create layer_classifier
    feat_extr = CNormalizerDNN(dnn, out_layer='features:relu4')
    # Compose classifier
    tsne = CReducerPTSNE(epochs=500,
                         batch_size=32,
                         preprocess=feat_extr,
                         random_state=random_state)
    clf = CClassifierMulticlassOVA(classifier=CClassifierKDE, kernel=CKernelRBF(), preprocess=tsne)

    # Select 10K training data and 1K test data (sampling)
    tr_idxs = CArray.randsample(vl.X.shape[0], shape=N_TRAIN, random_state=random_state)
    tr_sample = vl[tr_idxs, :]
    ts_idxs = CArray.randsample(ts.X.shape[0], shape=N_TEST, random_state=random_state)
    ts_sample = ts[ts_idxs, :]

    # 2. Load attack points
    fmt_string = "dnn_wb_seval_it_%d.gz"
    eps = 4.0
    # Loop
    adv_ds = load_adv_ds(fmt_string, 0, eps)
    for i in range(1, 3):
        adv_ds = adv_ds.append(load_adv_ds(fmt_string, i, eps))

    # Mark as adversarial samples
    # adv_ds.Y += tr_sample.num_classes                                 # a) classes > tr_sample.num_classes
    adv_ds.Y = CArray.ones(adv_ds.Y.shape[0]) * tr_sample.num_classes   # b) a single adversarial class

    # 3. Concatenate datasets
    tr_sample = tr_sample.append(adv_ds)

    # # Xval
    # def compute_hiddens(n_hiddens, n_layers):
    #     return sum([[[l] * k for l in n_hiddens] for k in range(1, n_layers+1)], [])
    #
    # xval_params = {
    #     'preprocess.preprocess.n_hiddens': compute_hiddens([64, 128, 256], 2),
    #     'kernel.gamma': [0.1, 1, 10, 100]
    # }
    #
    # # Let's create a 3-Fold data splitter
    # from secml.data.splitter import CDataSplitterKFold
    #
    # xval_splitter = CDataSplitterKFold(num_folds=3, random_state=random_state)
    #
    # # Parallel?
    # clf.n_jobs = 16
    #
    # # Select and set the best training parameters for the classifier
    # clf.verbose = 1
    # print("Estimating the best training parameters...")
    # best_params = clf.estimate_parameters(
    #     dataset=tr_sample,
    #     parameters=xval_params,
    #     splitter=xval_splitter,
    #     metric='accuracy',
    #     perf_evaluator='xval'
    # )
    #
    # print("The best training parameters are: ",
    #       [(k, best_params[k]) for k in sorted(best_params)])

    # Setting best params (external xval)
    clf.set_params({
        'kernel.gamma': 0.01,
        'preprocess.n_hiddens': [128, 128]}
    )
    # Expected performance: 0.9735

    # We can now create a classifier with reject
    clf.preprocess = None  # TODO: "preprocess should be passed to outer classifier..."
    clf_rej = CClassifierRejectThreshold(clf, -1000., preprocess=tsne)

    # We can now fit the clf_rej
    clf_rej.preprocess.verbose = 1  # DEBUG
    clf_rej.fit(tr_sample.X, tr_sample.Y)

    # Set threshold (FPR: 10%)
    clf_rej.threshold = clf_rej.compute_threshold(0.1, ts_sample)

    # Check test performance
    y_pred = clf_rej.predict(ts.X, return_decision_function=False)

    from secml.ml.peval.metrics import CMetric, CMetricAccuracy

    acc = CMetric.create('accuracy').performance_score(ts.Y, y_pred)
    print("Model Accuracy: {}".format(acc))

    # Dump to disk
    clf_rej.save('tsne_rej')
