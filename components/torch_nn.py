# from setGPU import setGPU
# setGPU(-1)

from secml.ml.peval.metrics import CMetricAccuracy
from torch import nn, optim
from collections import OrderedDict

from secml.array import CArray
from secml.data.loader import CDLRandom
from secml.data.splitter import CTrainTestSplit, CDataSplitterKFold
from secml.ml import CNormalizerMinMax, CClassifierPyTorch


class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens, n_class):
        super(MLP, self).__init__()
        assert isinstance(input_dims, int), 'Please provide int for input_dims'
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers['fc{}'.format(i + 1)] = nn.Linear(current_dims, n_hidden)
            layers['relu{}'.format(i + 1)] = nn.ReLU()
            layers['drop{}'.format(i + 1)] = nn.Dropout(0.2)
            current_dims = n_hidden
        layers['out'] = nn.Linear(current_dims, n_class)

        self.model = nn.Sequential(layers)
        print(self.model)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        return self.model.forward(input)


def make_layers(input_dims, n_hiddens, n_class):
    layers = OrderedDict()
    if isinstance(n_hiddens, int):
        n_hiddens = [n_hiddens]
    else:
        n_hiddens = list(n_hiddens)

    current_dims = input_dims
    for i, n_hidden in enumerate(n_hiddens):
        layers['fc{}'.format(i + 1)] = nn.Linear(current_dims, n_hidden)
        layers['relu{}'.format(i + 1)] = nn.ReLU()
        layers['drop{}'.format(i + 1)] = nn.Dropout(0.2)
        current_dims = n_hidden
    layers['out'] = nn.Linear(current_dims, n_class)

    return layers


class MLPytorch(CClassifierPyTorch):
    def __init__(self, input_dims, n_hiddens, n_class,
                 loss=None,
                 random_state=None, preprocess=None,
                 epochs=10, batch_size=1, lr=1e-3):
        self._input_dims = input_dims
        self._n_hiddens = n_hiddens
        self._n_class = n_class
        # Create DNN
        model = self._build_clf()

        # Loss and optimizer
        _loss = loss if loss is not None else nn.CrossEntropyLoss()
        self._lr = lr
        _optimizer = optim.SGD(model.parameters(), lr=self._lr)

        super().__init__(model, loss=_loss, optimizer=_optimizer, input_shape=(self._input_dims,),
                         random_state=random_state, preprocess=preprocess,
                         epochs=epochs, batch_size=batch_size)

    def _build_clf(self):
        layers = make_layers(self._input_dims, self._n_hiddens, self._n_class)
        return nn.Sequential(layers)

    @property
    def n_hiddens(self):
        return self._n_hiddens

    @n_hiddens.setter
    def n_hiddens(self, value):
        self._n_hiddens = value
        # Internals
        self._model = self._build_clf()
        self._init_model()
        self._optimizer = optim.SGD(self._model.parameters(), lr=self._lr)

    def _check_input(self, x, y=None):
        return x, y

    def _fit(self, x, y):
        # Storing dataset classes
        # TODO: CHECK POSSIBLE NON-FLAT SHAPES!
        self._n_features = x.shape[1]
        self._classes = CArray.arange(y.shape[1])

        return super()._fit(x, y)


if __name__ == '__main__':
    seed = 999

    ds = CDLRandom(n_samples=1000, n_features=20).load()

    # Split in training and test
    splitter = CTrainTestSplit(train_size=700,
                               test_size=300,
                               random_state=seed)
    tr, ts = splitter.split(ds)

    nmz = CNormalizerMinMax()
    tr.X = nmz.fit_transform(tr.X)
    ts.X = nmz.transform(ts.X)

    clf = MLPytorch(tr.X.shape[1], 100, ds.classes.size,
                    epochs=100,
                    batch_size=64,
                    random_state=seed)

    clf.verbose = 1

    # Xval
    xval_splitter = CDataSplitterKFold(num_folds=3, random_state=seed)
    params_grid = sum([[[l] * k for l in [8, 16, 32, 64, 128]] for k in range(1, 2)], [])
    best_params = clf.estimate_parameters(tr,
                                          parameters={'n_hiddens': params_grid},
                                          splitter=xval_splitter,
                                          metric='accuracy')
    print("The best training parameters are: ",
          [(k, best_params[k]) for k in sorted(best_params)])

    # Retrain classifier
    clf.fit(tr.X, tr.Y)

    # Predict
    y_pred = clf.predict(ts.X)
    acc = CMetricAccuracy().performance_score(ts.Y, y_pred)
    print("Accuracy of PyTorch Model: {:}".format(acc))

    print("done?")