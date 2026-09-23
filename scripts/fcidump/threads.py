"""Thread cap for pyscf scripts on a CPU-limited slice. Import BEFORE numpy/pyscf.

On the HELIX login node nproc reports 128 while the user slice is capped at 4 CPUs
(cgroup-v1 CFS quota). numpy's BLAS and pyscf's OpenMP each default to one thread per
visible core, which oversubscribes 32x: a He-atom SCF took 10-20 s instead of 0.05 s.
BLAS sizes its pool when it is loaded, so the environment has to be set before the first
numpy import; setting pyscf.lib.num_threads afterwards is not enough.

OMP_NUM_THREADS, when already set (e.g. by an sbatch script), wins.
"""

import os
import sys

if "numpy" in sys.modules and not os.environ.get("OMP_NUM_THREADS"):
    # BLAS has already sized its thread pool; setting the environment now would
    # silently leave it at one thread per visible core (131 threads, 7.8 s per FCI
    # sigma on a 4-CPU slice, measured). Fail instead of running oversubscribed. A
    # caller that set OMP_NUM_THREADS before loading numpy (the sbatch scripts,
    # dump_fcidump under them) has already capped the pools and may proceed.
    raise RuntimeError("threads must be imported before numpy, or "
                       "OMP_NUM_THREADS set before the process starts")


def cgroup_cpu_limit():
    """CPUs this process may actually use: the tightest cgroup-v1 CFS quota on the path
    from its cpu cgroup to the root, else os.cpu_count()."""
    best = None
    try:
        with open("/proc/self/cgroup") as fh:
            for line in fh:
                _, ctrl, path = line.rstrip("\n").split(":", 2)
                if "cpu" not in ctrl.split(","):
                    continue
                root = "/sys/fs/cgroup/" + ctrl
                d = root + path
                while True:
                    try:
                        q = int(open(d + "/cpu.cfs_quota_us").read())
                        per = int(open(d + "/cpu.cfs_period_us").read())
                        if q > 0 and per > 0:
                            n = max(1, q // per)
                            best = n if best is None else min(best, n)
                    except OSError:
                        pass
                    if d == root:
                        break
                    d = os.path.dirname(d)
    except OSError:
        pass
    return best or os.cpu_count() or 1


NTHREADS = int(os.environ.get("OMP_NUM_THREADS", "0")) or cgroup_cpu_limit()
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, str(NTHREADS))
