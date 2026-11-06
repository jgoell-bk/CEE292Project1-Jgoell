"""
Project 1 v2: DeepXDE soft vs. hard boundary conditions for Kovasznay flow (Jacob Goell)

Batch version of Project1v2.ipynb. The model, training and evaluation are the notebook's.
What is new here:
  - Every (method, seed) run trains at the same time, one process each, and the CPU
    cores are split evenly between them. Each run's normal DeepXDE output goes to its own
    log file.
  - Everything needed for post-processing is saved to disk (see Outputs below).

Usage (from the CEE292HW folder):
    sbatch project1v2.slurm             # Soft BC and Hard BC for every seed in SEEDS
    python project1v2.py --seeds 0      # seed 0 only, like the notebook
    python project1v2.py --quick        # 200 Adam + 1000 L-BFGS steps, seed 0: checks the pipeline

Outputs go to results_v2/run_<Slurm job id>/ (results_v2/run_local/ outside Slurm):
    config.json                       settings, package versions, node
    summary.csv                       one row per run: errors, residual, BC error, times, steps
    summary.txt                       mean ± std over seeds for each method
    runs/<method>_seed<k>.log         DeepXDE's printed output for that run
    runs/<method>_seed<k>_weights.pt  trained network weights (see load_model below)
    runs/<method>_seed<k>.npz         arrays for post-processing, with <method> = soft or hard:
        loss_steps (n,)                     steps at which DeepXDE recorded the loss: every
                                            DISPLAY_EVERY Adam steps, every 1000 L-BFGS iterations
        loss_train, loss_test (n, terms)    loss terms at those steps, named in loss_names
        train_points (3001, 2)              collocation + boundary points used for training
        test_points (100000, 2)             the notebook's evaluation points X
        test_pred, test_exact (100000, 3)   u, v, p
        test_residual (100000, 3)           momentum_x, momentum_y, continuity residuals
        grid_x (151,), grid_y (201,)        regular plotting grid with spacing 0.01
        grid_pred, grid_exact (201, 151, 3) u, v, p on the grid ([..., 0] is u)
        grid_residual (201, 151, 3)         the three residuals on the grid
        grid_gradients (201, 151, 6)        u_x, u_y, v_x, v_y, p_x, p_y on the grid
        wall_points (~4000, 2)              evenly spaced points on the four walls
        wall_pred, wall_exact (~4000, 3)    u, v, p on the walls

Post-processing example, in a notebook in the CEE292HW folder:
    import numpy as np
    import project1v2 as p1
    run = np.load("results_v2/run_<job id>/runs/hard_seed0.npz")
    u_error = run["grid_pred"][..., 0] - run["grid_exact"][..., 0]       # (201, 151)
    model = p1.load_model("results_v2/run_<job id>/runs/hard_seed0_weights.pt", hard_bc=True)
    uvp = model.predict(points)                       # the trained model at any (N, 2) points
    residuals = model.predict(points, operator=p1.pde)
"""
import argparse
import contextlib
import csv
import json
import multiprocessing as mp
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("DDE_BACKEND", "pytorch")   # NEW: pick the backend before deepxde is imported

import deepxde as dde
import numpy as np
from deepxde import backend as bkd
import torch


# Original flow parameters and domain
Re = 20     #initialize reynolds number
nu = 1 / Re
l = 1 / (2 * nu) - np.sqrt(1 / (4 * nu**2) + 4 * np.pi**2)      #Define lambda

x_min, x_max = -0.5, 1.0        #Set domain bounds
y_min, y_max = -0.5, 1.5


# NEW: run settings (the defaults are the notebook's)
ADAM_ITERATIONS = 30000
LBFGS_MAXITER = 15000                # DeepXDE's default; L-BFGS normally stops on its own before this.
                                     # DeepXDE runs L-BFGS in chunks of 1000 iterations, so use a multiple of 1000.
DISPLAY_EVERY = 1000                 # how often the loss history is recorded during Adam (DeepXDE's default).
                                     # Smaller gives a finer history but evaluates the 100,000 test points more often.
TRAIN_DISTRIBUTION = "Hammersley"    # DeepXDE's default. Hammersley points are the same for every seed, so
                                     # seeds only change the initial weights; "pseudo" also redraws the points.
SEEDS = [0, 1, 2, 3, 4]
METHODS = {"soft": False, "hard": True}   # file name tag -> hard_bc


# Original Navier–Stokes residuals
def pde(x, u):
    u_vel, v_vel = u[:, 0:1], u[:, 1:2]     #Extract velocity components

    u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)     #Calculate du/dx
    u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)     #Calculate du/dy
    u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)    #Calculate d2u/dx2
    u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)    #Calculate d2u/dy2

    v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)     #Same as above, just for v
    v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
    v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
    v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

    p_x = dde.grad.jacobian(u, x, i=2, j=0)         #Calculate pressure gradients
    p_y = dde.grad.jacobian(u, x, i=2, j=1)

    momentum_x = (u_vel * u_vel_x + v_vel * u_vel_y + p_x - nu * (u_vel_xx + u_vel_yy)) #Formulate x momentum eq
    momentum_y = (u_vel * v_vel_x + v_vel * v_vel_y + p_y - nu * (v_vel_xx + v_vel_yy)) #formulate y momentum eq
    continuity = u_vel_x + v_vel_y                                                      #Formulate continuity eq

    return [momentum_x, momentum_y, continuity]


# NEW: first derivatives of u, v, p, for post-processing (vorticity, strain rate, ...)
def gradients(x, u):
    # u_x, u_y, v_x, v_y, p_x, p_y
    return [dde.grad.jacobian(u, x, i=i, j=j) for i in range(3) for j in range(2)]


# Exact solution, used for evaluation
def u_func(x):
    return 1 - np.exp(l * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2])    #Exact solution for u, v and p

def v_func(x):
    return (l / (2 * np.pi) * np.exp(l * x[:, 0:1]) * np.sin(2 * np.pi * x[:, 1:2]))

def p_func(x):
    return 0.5 * (1 - np.exp(2 * l * x[:, 0:1]))

def exact_uvp(x):
    # NEW: u, v, p side by side, shape (N, 3)
    return np.hstack([u_func(x), v_func(x), p_func(x)])


# NEW: hard boundary transformation. Justification for G and D in report
def hard_boundary_transform(xy, raw_output):
    x, y = xy[:, 0:1], xy[:, 1:2] #Extract coordinates

    s = (x - x_min) / (x_max - x_min) # Linear interpolation
    A = (1 - s) * np.exp(l * x_min) + s * np.exp(l * x_max)

    # Fixed fields that match velocity
    G_u = 1 + bkd.exp(l * x) - (1 + bkd.cos(2 * np.pi * y)) * A
    G_v = l / (2 * np.pi) * A * bkd.sin(2 * np.pi * y)

    # Network corrections vanish on every edge
    D = (x - x_min) * (x_max - x) * (y - y_min) * (y_max - y)

    u = G_u + D * raw_output[:, 0:1] #Get u and v
    v = G_v + D * raw_output[:, 1:2]

    # Pressure condition only at outlet.
    p_outlet = 0.5 * (1 - np.exp(2 * l * x_max))
    p = p_outlet + (x_max - x) * raw_output[:, 2:3]

    return bkd.concat([u, v, p], axis=1)


spatial_domain = dde.geometry.Rectangle(xmin=[x_min, y_min], xmax=[x_max, y_max]) #Defines domain to train on

# Shared evaluation points for both models
dde.config.set_random_seed(1234)
X = spatial_domain.random_points(100000)

# NEW: regular grid and wall points, also shared by every run, for post-processing
grid_x = np.linspace(x_min, x_max, 151)
grid_y = np.linspace(y_min, y_max, 201)
grid_X, grid_Y = np.meshgrid(grid_x, grid_y)   # shape (201, 151): rows are y, columns are x
grid_points = np.stack([grid_X.ravel(), grid_Y.ravel()], axis=1).astype(np.float32)
wall_points = spatial_domain.uniform_boundary_points(4000)


def build_model(hard_bc):
    # The notebook's data, network and hard constraint. Used for training and by load_model.
    if hard_bc:
        # CHANGED: no boundary penalties
        boundary_conditions = [] #No longer boundary penalties w/ hard constraint

    else:
        # Original soft boundary conditions from the documentation
        boundary_u = dde.icbc.DirichletBC(
            spatial_domain,
            u_func,
            lambda _, on_boundary: on_boundary,
            component=0,
        )

        boundary_v = dde.icbc.DirichletBC(
            spatial_domain,
            v_func,
            lambda _, on_boundary: on_boundary,
            component=1,
        )

        boundary_p = dde.icbc.DirichletBC(
            spatial_domain,
            p_func,
            lambda x, on_boundary: (
                on_boundary and dde.utils.isclose(x[0], x_max)
            ),
            component=2,
        )

        boundary_conditions = [boundary_u, boundary_v, boundary_p]

    data = dde.data.PDE(
        spatial_domain,
        pde,
        boundary_conditions,
        num_domain=2601,
        num_boundary=400,  # These points still contribute to the PDE residual
        train_distribution=TRAIN_DISTRIBUTION,
        num_test=100000,
    )

    net = dde.nn.FNN([2] + 4 * [50] + [3], "tanh", "Glorot normal")

    if hard_bc:
        # NEW: apply hard constraint
        net.apply_output_transform(hard_boundary_transform)

    return dde.Model(data, net) #Initialize model


def load_model(weights_path, hard_bc):
    # NEW: rebuild a trained model from its saved weights, for post-processing.
    # model.predict(points) and model.predict(points, operator=pde) then work as in the notebook.
    model = build_model(hard_bc)
    model.compile("adam", lr=1e-3)   # predict needs a compiled model; the optimizer is not used
    model.net.load_state_dict(torch.load(weights_path))
    return model


def train_and_evaluate(hard_bc=False, seed=0, run_dir=Path("."), adam_iterations=ADAM_ITERATIONS,
                       lbfgs_maxiter=LBFGS_MAXITER, display_every=DISPLAY_EVERY):
    name = "Hard BC" if hard_bc else "Soft BC"
    tag = "hard" if hard_bc else "soft"
    print(f"\nStarting {name}, Re = {Re}, seed = {seed}")

    # Reset TensorFlow state before creating each fresh model
    if dde.backend.backend_name == "tensorflow.compat.v1":
        bkd.tf.reset_default_graph()
    elif dde.backend.backend_name == "tensorflow":
        bkd.tf.keras.backend.clear_session()

    # Use the same seed for sampling and network initialization
    dde.config.set_random_seed(seed)

    if dde.backend.backend_name == "tensorflow":
        bkd.tf.keras.utils.set_random_seed(seed)

    model = build_model(hard_bc)

    # Training
    start = time.perf_counter()

    model.compile("adam", lr=1e-3)
    model.train(iterations=adam_iterations, display_every=display_every)
    adam_time = time.perf_counter() - start
    adam_steps = model.train_state.step

    dde.optimizers.set_LBFGS_options(maxiter=lbfgs_maxiter)
    model.compile("L-BFGS") #So the original formulation switches to a L-BFGS optimizer after 30000 iterations. It shuts off on it's own. Didn't look into this deeply but it's from the original implementation.
    losshistory, train_state = model.train()

    training_time = time.perf_counter() - start

    # Original evaluation
    output = model.predict(X)

    #Output predictions
    u_pred = output[:, 0]
    v_pred = output[:, 1]
    p_pred = output[:, 2]

    #Grab analytical solution
    u_exact = u_func(X).reshape(-1)
    v_exact = v_func(X).reshape(-1)
    p_exact = p_func(X).reshape(-1)

    f = model.predict(X, operator=pde) #Evaluate residuals from pde

    l2_difference_u = dde.metrics.l2_relative_error(u_exact, u_pred) #Calculates error against exact solution.
    l2_difference_v = dde.metrics.l2_relative_error(v_exact, v_pred)
    l2_difference_p = dde.metrics.l2_relative_error(p_exact, p_pred)

    residual = np.mean(np.absolute(f))

    # NEW: predictions on the grid and the walls
    wall_pred = model.predict(wall_points)
    wall_exact = exact_uvp(wall_points)
    bc_error_u = np.max(np.abs(wall_pred[:, 0] - wall_exact[:, 0]))   # largest u, v mismatch on the walls
    bc_error_v = np.max(np.abs(wall_pred[:, 1] - wall_exact[:, 1]))

    grid_shape = grid_X.shape + (-1,)   # (201, 151, -1)
    grid_pred = model.predict(grid_points).reshape(grid_shape)
    grid_residual = np.hstack(model.predict(grid_points, operator=pde)).reshape(grid_shape)
    grid_gradients = np.hstack(model.predict(grid_points, operator=gradients)).reshape(grid_shape)

    print(f"\n{name} results")
    print("Mean residual:", residual)
    print("L2 relative error in u:", l2_difference_u)
    print("L2 relative error in v:", l2_difference_v)
    print("L2 relative error in p:", l2_difference_p)
    print("Training time (s):", training_time)

    # NEW: save everything needed for post-processing
    loss_names = ["momentum_x", "momentum_y", "continuity"] + ([] if hard_bc else ["bc_u", "bc_v", "bc_p"])
    np.savez(
        run_dir / f"{tag}_seed{seed}.npz",
        loss_steps=np.array(losshistory.steps),
        loss_train=np.array(losshistory.loss_train),
        loss_test=np.array(losshistory.loss_test),
        loss_names=np.array(loss_names),
        train_points=model.data.train_x_all,
        test_points=X,
        test_pred=output,
        test_exact=exact_uvp(X),
        test_residual=np.hstack(f),
        grid_x=grid_x,
        grid_y=grid_y,
        grid_pred=grid_pred,
        grid_exact=exact_uvp(grid_points).reshape(grid_shape),
        grid_residual=grid_residual,
        grid_gradients=grid_gradients,
        wall_points=wall_points,
        wall_pred=wall_pred,
        wall_exact=wall_exact,
    )
    torch.save(model.net.state_dict(), run_dir / f"{tag}_seed{seed}_weights.pt")

    return {
        "method": name,
        "seed": seed,
        "u_error": float(l2_difference_u),
        "v_error": float(l2_difference_v),
        "p_error": float(l2_difference_p),
        "residual": float(residual),
        "bc_error_u": float(bc_error_u),
        "bc_error_v": float(bc_error_v),
        "training_time": training_time,
        "adam_time": adam_time,
        "lbfgs_time": training_time - adam_time,
        "adam_steps": adam_steps,
        "lbfgs_steps": train_state.step - adam_steps,
        "final_train_loss": float(np.sum(losshistory.loss_train[-1])),
        "final_test_loss": float(np.sum(losshistory.loss_test[-1])),
    }


# ---------------------------------------------------------------------------
# NEW: run every (method, seed) at the same time, one process each
# ---------------------------------------------------------------------------
def init_worker(threads):
    # Each run process uses only its own share of the cores
    torch.set_num_threads(threads)


def run_one(task):
    # One training run, in its own process. DeepXDE's printed output goes to the run's log file.
    tag, seed, run_dir, settings = task
    with open(run_dir / f"{tag}_seed{seed}.log", "w", buffering=1) as log, \
            contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        return train_and_evaluate(hard_bc=METHODS[tag], seed=seed, run_dir=run_dir, **settings)


def write_summary_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_table(rows, path):
    # Mean ± std over seeds for each method
    columns = ["u_error", "v_error", "p_error", "residual", "bc_error_u", "bc_error_v", "training_time", "lbfgs_steps"]
    lines = [f"{'Method':<10}" + "".join(f"{c:>22}" for c in columns)]
    for method in ["Soft BC", "Hard BC"]:
        runs = [row for row in rows if row["method"] == method]
        if not runs:
            continue
        line = f"{method:<10}"
        for c in columns:
            values = [run[c] for run in runs]
            line += f"{np.mean(values):>12.3e} ± {np.std(values):<7.1e}"
        lines.append(line)
    table = "\n".join(lines) + f"\n({len(rows)} runs; seeds: {sorted({row['seed'] for row in rows})})\n"
    print(table)
    Path(path).write_text(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help=f"seeds to run (default: {SEEDS})")
    parser.add_argument("--quick", action="store_true",
                        help="200 Adam steps + 1000 L-BFGS iterations and seed 0 only, to check the pipeline")
    args = parser.parse_args()

    settings = dict(adam_iterations=ADAM_ITERATIONS, lbfgs_maxiter=LBFGS_MAXITER, display_every=DISPLAY_EVERY)
    seeds = args.seeds
    if args.quick:
        settings = dict(adam_iterations=200, lbfgs_maxiter=1000, display_every=100)
        seeds = [0]

    out_dir = Path("results_v2") / f"run_{os.environ.get('SLURM_JOB_ID', 'local')}"
    run_dir = out_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    # One run per (method, seed), all at the same time. The cores are split evenly between them.
    tasks = [(tag, seed, run_dir, settings) for tag in METHODS for seed in seeds]
    n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    n_workers = min(len(tasks), n_cpus)
    threads_per_run = max(1, n_cpus // n_workers)

    config = {
        "Re": Re, "lambda": l, "domain": [[x_min, x_max], [y_min, y_max]],
        "network": "FNN [2] + 4 * [50] + [3], tanh, Glorot normal",
        "num_domain": 2601, "num_boundary": 400, "num_test": 100000, "train_distribution": TRAIN_DISTRIBUTION,
        "adam_lr": 1e-3, **settings, "seeds": seeds, "methods": list(METHODS),
        "deepxde": dde.__version__, "torch": torch.__version__, "numpy": np.__version__,
        "backend": dde.backend.backend_name, "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "cpus": n_cpus, "threads_per_run": threads_per_run,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"{len(tasks)} runs, {n_workers} at a time with {threads_per_run} threads each "
          f"({n_cpus} cores). Output: {out_dir}", flush=True)

    rows = []
    with mp.get_context("spawn").Pool(n_workers, initializer=init_worker, initargs=(threads_per_run,)) as pool:
        for row in pool.imap_unordered(run_one, tasks):
            print(f"[{row['method']}, seed {row['seed']}] done: u {row['u_error']:.2e}, v {row['v_error']:.2e}, "
                  f"p {row['p_error']:.2e}, {row['lbfgs_steps']} L-BFGS steps, {row['training_time']:.0f} s", flush=True)
            rows.append(row)
            write_summary_csv(rows, out_dir / "summary.csv")   # after every run, so finished runs survive a crash

    rows.sort(key=lambda row: (row["method"], row["seed"]))
    write_summary_csv(rows, out_dir / "summary.csv")
    write_summary_table(rows, out_dir / "summary.txt")
    print(f"Done. Results are in {out_dir}", flush=True)


if __name__ == "__main__":
    main()
