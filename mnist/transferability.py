from secml.adv.seceval import CSecEval
from secml.ml.classifiers.reject import CClassifierRejectThreshold, CClassifierDNR


def transfer_attack(clf, seval):
    '''
    Performs transferability of seval to clf classifier
    :param clf: Classifier to be tested for transfer attack
    :param seval: Security evaluation containing attack data
    :return: seval for tranferability attack on clf
    '''

    res = seval.copy()

    # TODO: Remove unnecessary params
    # res.sec_eval_data.fobj = None

    # Main loop
    for i, ds in enumerate(res.sec_eval_data.adv_ds):
        pred, scores = clf.predict(ds.X, return_decision_function=True)
        res.sec_eval_data.scores[i] = scores
        # TODO: CHECK SCORE FOR NATURAL CLASSES
        res.sec_eval_data.Y_pred[i] = pred

    return res


# from mnist.tsne_rej_gamma_test import GAMMA
# CLFS = ['tsne_rej_test_gamma_' + str(gamma) for gamma in GAMMA]
CLFS = ['tsne_rej', 'tnr']     # 'nr', 'dnr',
if __name__ == '__main__':
    random_state = 999

    # Load adversarial samples
    dnn_seval = CSecEval.load("dnn_seval.gz")

    for _clf in CLFS:

        # Load clf
        if _clf == 'nr' or _clf == 'tsne_rej':
            clf = CClassifierRejectThreshold.load(_clf + '.gz')
        elif _clf == 'dnr' or _clf == 'tnr':
            clf = CClassifierDNR.load(_clf + '.gz')
        else:
            raise ValueError("Unknown model to test for transferability!")
        clf.n_jobs = 16

        print("- Transfer to ", _clf)

        # Transferability test
        transfer_seval = transfer_attack(clf, dnn_seval)

        # Dump to disk
        transfer_seval.save(_clf + "_bb_seval")
