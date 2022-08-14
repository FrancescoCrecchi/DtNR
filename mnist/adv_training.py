# Basic separability test for DNN WB attack samples

from secml.adv.attacks import CAttackEvasionPGD
from secml.array import CArray
from secml.data import CDataset
from secml.figure import CFigure
from secml.ml import CNormalizerDNN, CKernelRBF, CNormalizerMinMax
from secml.ml.classifiers.multiclass import CClassifierMulticlassOVA
from secml.ml.peval.metrics import CMetricAccuracy
from sklearn.manifold import TSNE

from components.c_classifier_kde import CClassifierKDE
from components.c_reducer_ptsne import CReducerPTSNE
from mnist.cnn_mnist import cnn_mnist_model
from mnist.fit_dnn import get_datasets

DSET = 'mnist'
N_SAMPLES = 200
MARGIN = 50
EPS = 1.0

# Load 1000 samples
random_state = 999
_, vl, ts = get_datasets(random_state)

# Load classifier
dnn = cnn_mnist_model()
dnn.load_model('cnn_mnist.pkl')
# Wrap it in a CNormalizerDNN
feat_extr = CNormalizerDNN(dnn, out_layer='features:relu4')

# Select a sample from dset
sample = vl[:N_SAMPLES+MARGIN, :]  # 300 is a margin for non-evading samples

# Load attack
pgd_attack = CAttackEvasionPGD.load('dnn_attack.gz')
# Resetting classifier and surrogate
pgd_attack._classifier = dnn
pgd_attack.surrogate_classifier = dnn
# Setting max distance
pgd_attack.dmax = EPS

# Run evasion
pgd_attack.verbose = 2 # DEBUG
eva_y_pred, _, eva_adv_ds, _ = pgd_attack.run(sample.X, sample.Y)

# Select effective evading samples
evading_samples = eva_adv_ds[eva_y_pred != sample.Y, :]
N = min(evading_samples.X.shape[0], N_SAMPLES)
evading_samples = evading_samples[:N, :]

X_nat, y_nat = sample[:N, :].X, sample[:N, :].Y + 1
X_adv, y_adv = evading_samples.X, -(evading_samples.Y + 1)    # Negating to differentiate natural from adv. samples

# Pass through features extractor
# X_embds = feat_extr.transform(CArray.concatenate(X_nat, X_adv, axis=0))
X_embds = feat_extr.transform(X_nat)

# TSNE part
# X_2d = CArray(TSNE(verbose=1).fit_transform(X_embds.tondarray()))
tsne = CReducerPTSNE(epochs=250,
                     batch_size=128,
                     random_state=random_state,
                     n_jobs=10) # n_hiddens=[256, 128, 64], epochs=1000, batch_size=32
tsne.verbose = 1
X_2d = tsne.fit_transform(X_embds)
# y_2d = CArray.concatenate(y_nat, y_adv)
y_2d = y_nat

# Kde classifier ontop to inspect scores
kde = CClassifierMulticlassOVA(classifier=CClassifierKDE, kernel=CKernelRBF(gamma=0.01))
bool_idx = y_2d > 0
kde.fit(X_2d[bool_idx, :], y_2d[bool_idx])

# Check for predictive performance
y_pred = kde.predict(X_2d[bool_idx, :]) + 1 # as above..
acc = CMetricAccuracy().performance_score(y_2d[bool_idx], y_pred)
print("Predictive performance on TRAINING data: {0:.2f}".format(acc))

# Embedding new points
test_sample = ts[:N_SAMPLES, :]
X_2d_ts, y_2d_ts = tsne.transform(feat_extr.transform(test_sample.X)), test_sample.Y + 1

# Check for predictive performance
y_pred = kde.predict(X_2d_ts) + 1 # as above..
acc = CMetricAccuracy().performance_score(y_2d_ts, y_pred)
print("Predictive performance on TEST data: {0:.2f}".format(acc))

# Plot separability with TSNE
train = CDataset(X_2d, y_2d)
test = CDataset(X_2d_ts, y_2d_ts)

fig = CFigure(height=8, width=10)
fig.sp.plot_decision_regions(kde, grid_limits=[(-120, 120), (-120, 120)])
fig.sp.plot_ds(train, alpha=0.5)
fig.sp.plot_ds(test)
# REF_CLASS = 5
# foo_ref = foo[(foo.Y == REF_CLASS+1).logical_or(foo.Y == -(REF_CLASS+1)), :]
# foo_ref = foo[(foo.Y == REF_CLASS+1).logical_or(foo.Y < 0), :]
# 1vsRest
# foo_ref = foo[(foo.Y == REF_CLASS+1).logical_or(foo.Y != REF_CLASS+1), :]
# foo_ref.Y[foo_ref.Y != REF_CLASS+1] = -1.0
# fig.sp.plot_ds(foo_ref)
fig.savefig('adv_train.png')

