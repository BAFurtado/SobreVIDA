"""Run independent model replications across processes.

Every sweep cell is a set of independent runs of one parameter vector, so nothing about
them is sequential. What keeps them serial is that a Home cannot travel back to the
parent: the mesa schedule holds agent references that do not pickle. Each worker
therefore runs the model AND reduces it to plain records before returning.

Measured cost of one 144-step PORTO ALEGRE run: ~80s on one core (7.5s to build the
population, 73s to step it). A 5-cell grid at 5 iterations is 25 runs, so ~33 min serial
against ~4 min on ten workers.

RESEEDING IS NOT OPTIONAL. Under fork a worker copies the RNG state of the parent, so
without this every replicate in a cell would draw the identical stream and the cell would
report one run repeated N times with zero dispersion. Two generators need it, because the
model uses both: numpy's global (model.py draws thresholds, partner turnover and network
weights from it) and the stdlib global (mesa's Model.__new__ takes its own per-model seed
from random.random(), see mesa/model.py:58).

START METHOD. fork where the platform has it, spawn where it does not (Windows has no
fork). Results do not depend on it:

  - Nothing depends on inheriting parameters. Every harness sets its cell's parameters
    inside the worker entry point (action_sweep._one_run, prevalence_sweep._one_run,
    persistence_diagnostic._one_run, scenarios.run_once), because pool workers are reused
    ACROSS cells even under fork.
  - Nothing depends on inheriting the RNG either: the initializer below reseeds, and the
    seeded harnesses pass an explicit per-replicate seed in the job itself, which makes
    the common-random-number pairing reproducible on any platform.

What spawn does cost is startup: each worker re-imports pandas, mesa and the model, and
re-reads the census tables that input/population.py loads at import. That is paid once per
worker per pool, and a pool is built once per cell. Set SOBREVIDA_START_METHOD to force a
particular method.
"""
import os
import random
import sys
from multiprocessing import get_context

sys.path.append(os.path.abspath('violence'))

# Leave a couple of cores for the OS and for whatever else the machine is doing; the
# workers are CPU-bound for their whole life.
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 2)


def pool_context():
    """The multiprocessing context to build pools from — see the module docstring.

    Preference order: whatever SOBREVIDA_START_METHOD names, then fork, then spawn. Each
    is tried rather than tested, because availability is exactly what get_context reports.
    """
    requested = os.environ.get('SOBREVIDA_START_METHOD')
    for method in (m for m in (requested, 'fork', 'spawn') if m):
        try:
            return get_context(method)
        except ValueError:
            continue
    return get_context()


def _seed_worker():
    """Give this process an RNG state unrelated to the parent's and to its siblings."""
    seed = int.from_bytes(os.urandom(8), 'little')
    random.seed(seed)
    # Imported here rather than at module scope so that a serial caller who never starts
    # a pool does not pay for it.
    import numpy as np
    np.random.seed(seed % (2 ** 32))


def map_runs(task, jobs, workers=None, on_result=None):
    """Apply `task` to each entry of `jobs`, in parallel, and return results in order.

    `task` must be a module-level function (it is pickled by name) and must return
    something picklable — records and counters, never a Home or an agent.

    `workers=1` runs everything in this process instead of forking. Use it when a
    traceback from inside a worker would be hard to read, or when profiling.

    `on_result(index, result)` is called as each job lands, for progress reporting.
    Results arrive out of order, hence the index.
    """
    jobs = list(jobs)
    n = workers if workers is not None else DEFAULT_WORKERS
    n = max(1, min(n, len(jobs)))

    if n == 1:
        out = []
        for i, job in enumerate(jobs):
            r = task(job)
            if on_result:
                on_result(i, r)
            out.append(r)
        return out

    results = [None] * len(jobs)
    ctx = pool_context()
    with ctx.Pool(processes=n, initializer=_seed_worker) as pool:
        for i, r in pool.imap_unordered(_indexed(task), list(enumerate(jobs))):
            results[i] = r
            if on_result:
                on_result(i, r)
    return results


class _indexed:
    """Carry the job index through imap_unordered so results can be put back in order.

    A class rather than a closure because the callable is pickled to reach the worker.
    """

    def __init__(self, task):
        self.task = task

    def __call__(self, item):
        i, job = item
        return i, self.task(job)