import argparse
from pso import pso


if __name__ == "__main__":
    """Usage
    python3 main.py -n 100 -e 10 -s 1
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c1", "--c1", type=float, default=0.7184)
    parser.add_argument("-c2", "--c2", type=float, default=0.8032)
    parser.add_argument("-w", "--w", type=float, default=0.1450)
    parser.add_argument("-k", "--k", type=int, default=10)
    parser.add_argument("-p", "--p", type=int, default=2)
    parser.add_argument("-n", "--n-particles", type=int, default=10000)
    parser.add_argument("-e", "--epochs", type=int, default=10000)
    parser.add_argument("-m", "--mode", type=str, default="gbest")
    parser.add_argument("-s", "--search", type=int, default=0)
    parser.add_argument("-l", "--log-dir", type=str, default="./logs.json")
    args, unknown = parser.parse_known_args()

    c1 = args.c1
    c2 = args.c2
    w = args.w
    k = args.k
    p = args.p
    n_particles = args.n_particles
    epochs = args.epochs
    search = args.search
    log_dir = args.log_dir
    mode = args.mode.lower()

    if search:
        import os
        from bayes_opt import BayesianOptimization
        from bayes_opt.observer import JSONLogger
        from bayes_opt.event import Events
        from bayes_opt.util import load_logs

        pbounds = {"c1": (0, 1.0),
                   "c2": (0, 1.0),
                   "w": (0, 1.0)}
        optimizer = BayesianOptimization(f=pso, pbounds=pbounds)
        if os.path.exists(log_dir):
            load_logs(optimizer, logs=[log_dir])
        optimizer.subscribe(Events.OPTMIZATION_STEP, JSONLogger(path=log_dir))
        optimizer.maximize(init_points=100, n_iter=25)
        print(optimizer.max)
    else:
        pso(c1, c2, w, k, p, n_particles, epochs, mode, verbose=2, visualize=1)
