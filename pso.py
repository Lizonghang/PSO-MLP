import numpy as np
from sklearn.datasets import load_digits

# split dataset
DATA = load_digits()
X = DATA.data
Y = DATA.target
NUM_SAMPLES = 1797
NUM_TRAINSET = 1500
indices = list(range(NUM_SAMPLES))
np.random.shuffle(indices)
train_indices = indices[:NUM_TRAINSET]
test_indices = indices[NUM_TRAINSET:]
X_train = X[train_indices, :]
X_test = X[test_indices, :]
Y_train = Y[train_indices]
Y_test = Y[test_indices]

# data attributes
H = 8
W = 8
NUM_INPUTS = H * W
NUM_CLASSES = 10

# network layers
LAYERS = (NUM_INPUTS, 32, 32, NUM_CLASSES)
NUM_LAYERS = len(LAYERS) - 1
DIMENSIONS = 0
for l in range(NUM_LAYERS):
    DIMENSIONS += LAYERS[l] * LAYERS[l+1]
    DIMENSIONS += LAYERS[l+1]


def network(params, X):
    # construct layers from sequential params
    Ws = []
    bs = []
    begin = 0
    for l in range(NUM_LAYERS):
        num_weights = LAYERS[l] * LAYERS[l+1]
        num_biases = LAYERS[l+1]
        W = params[begin:begin+num_weights]\
            .reshape((LAYERS[l], LAYERS[l+1]))
        b = params[begin+num_weights:begin+num_weights+num_biases]\
            .reshape((LAYERS[l+1],))
        Ws.append(W)
        bs.append(b)
        begin += (num_weights + num_biases)
    # forward propagation
    h = X
    for l in range(NUM_LAYERS):
        if l == 0:
            h = np.tanh(X.dot(Ws[l]) + bs[l])
        elif l == NUM_LAYERS - 1:
            h = h.dot(Ws[l]) + bs[l]
        else:
            h = np.tanh(h.dot(Ws[l]) + bs[l])
    return h


def forward_prop(params):
    output = network(params, X_train)
    # softmax
    exp_scores = np.exp(output)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # cross entropy loss
    log_probs = -np.log(probs[range(NUM_TRAINSET), Y_train])
    loss = np.sum(log_probs) / NUM_TRAINSET
    return loss


def f(x):
    n_particles = x.shape[0]
    j = [forward_prop(x[i]) for i in range(n_particles)]
    return np.array(j)


def predict(X, params):
    output = network(params, X)
    y_pred = np.argmax(output, axis=1)
    return y_pred


def pso(c1, c2, w, k=2, p=2, n_particles=100, epochs=100, mode="gbest", verbose=0, visualize=0):
    options = {'c1': c1, 'c2': c2, 'w': w}

    # bound
    max_value = 1.0 * np.ones(DIMENSIONS)
    min_value = -1.0 * np.ones(DIMENSIONS)
    bounds = (min_value, max_value)

    # create pso optimizer
    if mode == "gbest":
        from pyswarms.single.global_best import GlobalBestPSO
        optimizer = GlobalBestPSO(n_particles=n_particles,
                                  dimensions=DIMENSIONS,
                                  options=options,
                                  bounds=bounds)
    elif mode == "lbest":
        from pyswarms.single.local_best import LocalBestPSO
        options.update({"k": k, "p": p})
        optimizer = LocalBestPSO(n_particles=n_particles,
                                 dimensions=DIMENSIONS,
                                 options=options,
                                 bounds=bounds)
    else:
        raise NotImplementedError("Mode %s not support." % mode)

    # optimize
    cost, params = optimizer.optimize(f, epochs, verbose=verbose)

    if verbose:
        print("Accuracy: ", (predict(X_test, params) == Y_test).mean())

    if visualize:
        import matplotlib.pyplot as plt
        from pyswarms.utils.plotters import plot_cost_history
        plot_cost_history(cost_history=optimizer.cost_history)
        plt.show()

    # for bayesian optimization
    return -cost
