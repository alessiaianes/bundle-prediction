

import logging

from dipy.utils.optpkg import optional_package
import numpy as np

import pickle

ray, has_ray, _ = optional_package("ray")


import os
from concurrent.futures import ProcessPoolExecutor


def furthest_first_traversal(S, k, distance, permutation=True, n_jobs=1, chunk_size=None):
    """Select prototypes with farthest-first traversal.

    This is a good sub-optimal solution to the k-center problem.
    See for example:
    Hochbaum, Dorit S. and Shmoys, David B., A Best Possible Heuristic
    for the k-Center Problem, Mathematics of Operations Research, 1985.
    or: http://en.wikipedia.org/wiki/Metric_k-center

    Compared to the naive implementation, this version maintains a running
    vector of minimum distances to the already-chosen prototypes, so each
    loop iteration only needs to compute distances to the single most recently
    added prototype rather than recomputing distances to every prototype chosen
    so far. That reduces the total distance-function work from O(k·n) calls
    to O(n) calls (k batches of n, but each batch touches only one new column).

    When ``n_jobs > 1`` the n-point distance computation inside each iteration
    is split across processes using ``ProcessPoolExecutor``. Each worker
    receives a contiguous chunk of ``S`` and the new prototype, computes
    distances independently, and returns a partial distance array. The main
    process then merges the partial results and updates ``min_dists``. This
    avoids GIL contention entirely and is appropriate when ``distance`` has
    meaningful Python-level overhead or when true CPU parallelism is needed.

    Note that ``distance`` and all arguments it closes over must be picklable,
    since they are serialised and sent to worker processes. Lambda functions
    and closures over unpicklable objects will raise at runtime.

    Parameters
    ----------
    S : ndarray
        Input samples to select from.
    k : int
        Number of samples to select.
    distance : callable
        Function that computes pairwise distances between a set of samples
        and a single reference point (or a small set of reference points).
        Must accept two positional arguments: a subset of rows from ``S``
        and a 2-D reference array, and return a 1-D array of distances.
        Must be picklable (i.e. a module-level function or a picklable
        callable), since it is sent to worker processes.
    permutation : bool, optional
        If True, permute ``S`` before selecting prototypes. Permuting removes
        any ordering bias in the input without changing the statistical
        properties of the result.
    n_jobs : int, optional
        Number of worker processes to use. Defaults to 1 (no
        multiprocessing). Set to -1 to use all available CPUs.
    chunk_size : int or None, optional
        Number of rows of ``S`` processed by each worker process. When None,
        a chunk size is chosen automatically as ``ceil(n / n_jobs)``.

    Returns
    -------
    ndarray
        Indices of the selected samples in the original (pre-permutation)
        input order.
    """
    n = S.shape[0]

    if permutation:
        idx = np.random.permutation(n)
        S = S[idx]
    else:
        idx = np.arange(n, dtype=np.int32)

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    min_dists = np.full(n, np.inf)

    T = [0]
    _update_min_dists(S, T[0], min_dists, distance, n_jobs, chunk_size)

    while len(T) < k:
        z = int(min_dists.argmax())
        T.append(z)
        _update_min_dists(S, z, min_dists, distance, n_jobs, chunk_size)
        print(len(T), end="\r")

    return idx[T]


def _compute_chunk_distances(args):
    """Compute distances from a chunk of samples to a single prototype.

    This function is the unit of work executed in each worker process. It is
    a plain module-level function rather than a closure or lambda so that
    Python's pickle machinery can serialise it for dispatch to worker
    processes. Closures and lambdas are not picklable and would raise a
    ``PicklingError`` at runtime.

    Parameters
    ----------
    args : tuple
        A tuple of ``(S_chunk, ref, distance, start)`` where:

        - ``S_chunk`` is the slice of the sample matrix assigned to this
          worker, with shape ``(chunk_size, n_features)``.
        - ``ref`` is the prototype row, with shape ``(1, n_features)``.
        - ``distance`` is the callable passed by the caller of
          ``furthest_first_traversal``. It must be picklable.
        - ``start`` is the row offset of this chunk in the full ``S`` matrix,
          returned unchanged so the main process can place results correctly.

    Returns
    -------
    tuple
        ``(start, distances)`` where ``distances`` is a 1-D array of length
        ``chunk_size`` giving the distance from each row of ``S_chunk`` to
        ``ref``.
    """
    S_chunk, ref, distance, start = args
    return start, distance(S_chunk, ref).ravel()


def _update_min_dists(S, new_prototype_idx, min_dists, distance, n_jobs, chunk_size):
    """Update ``min_dists`` in-place after adding a new prototype.

    Computes the distance from every point in ``S`` to the single new
    prototype, then takes the element-wise minimum with ``min_dists``.

    When ``n_jobs == 1``, the computation runs in the main process with no
    multiprocessing overhead. When ``n_jobs > 1``, ``S`` is split into
    contiguous chunks, each chunk is dispatched to a worker process, and the
    returned partial distance arrays are merged back in the main process.

    The merge step is a sequential ``np.minimum`` over the returned chunks.
    This is cheap relative to the distance computation itself because it
    operates on already-computed scalars rather than calling ``distance``.

    Parameters
    ----------
    S : ndarray
        Full sample matrix (already permuted if requested).
    new_prototype_idx : int
        Row index into ``S`` of the prototype just added.
    min_dists : ndarray
        Running per-point minimum distances; updated in-place.
    distance : callable
        Pairwise distance function. Must be picklable.
    n_jobs : int
        Number of worker processes.
    chunk_size : int or None
        Rows per worker chunk; inferred when None.
    """
    n = S.shape[0]
    ref = S[[new_prototype_idx]]  # shape (1, n_features)

    if n_jobs == 1:
        new_dists = distance(S, ref).ravel()
        np.minimum(min_dists, new_dists, out=min_dists)
        return

    effective_chunk = chunk_size or int(np.ceil(n / n_jobs))

    # Build the argument list for each worker. Each tuple is independently
    # picklable, which is required for ProcessPoolExecutor dispatch.
    chunk_args = [
        (S[start : start + effective_chunk], ref, distance, start)
        for start in range(0, n, effective_chunk)
    ]

    # ProcessPoolExecutor.map preserves submission order, but we use the
    # explicit `start` offset in each result to be defensive — if chunks
    # were ever reordered (e.g. by a different executor), placement would
    # still be correct.
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for start, chunk_dists in executor.map(_compute_chunk_distances, chunk_args):
            end = start + len(chunk_dists)
            np.minimum(min_dists[start:end], chunk_dists, out=min_dists[start:end])




def subset_furthest_first(S, k, distance, permutation=True, c=2.0):
    """Stochastic scalable version of the fft algorithm based in a
    random subset of a specific size.

    See: E. Olivetti, T.B. Nguyen, E. Garyfallidis, The Approximation
    of the Dissimilarity Projection, Proceedings of the 2012
    International Workshop on Pattern Recognition in NeuroImaging
    (PRNI), vol., no., pp.85,88, 2-4 July 2012 doi:
    10.1109/PRNI.2012.13

    D. Turnbull and C. Elkan, Fast Recognition of Musical Genres
    Using RBF Networks, IEEE Trans Knowl Data Eng, vol. 2005, no. 4,
    pp. 580-584, 17.
    """
    size = int(max(1, np.ceil(c * k * np.log(k))))
    if permutation:
        idx = np.random.permutation(S.shape[0])[:size]
    else:
        idx = range(size)
    # note: no need to add extra permutation here below:
    return idx[furthest_first_traversal(S[idx], k, distance, permutation=False)]


def compute_dissimilarity(
    data,
    distance,
    prototype_policy,
    num_prototypes,
    verbose=False,
    size_limit=5000000,
    n_jobs=6,
    sub=False
):
    """Compute dissimilarity matrix given data, distance,
    prototype_policy and number of prototypes.
    """
    logging.info("Computing dissimilarity matrix.")
    data_original = data
    num_proto = num_prototypes
    if data.shape[0] > size_limit:
        logging.info("Datset too big: subsampling to %s entries only!" % size_limit)
        data = data[np.random.permutation(data.shape[0])[:size_limit], :]

    logging.info("Number of prototypes: %s" % num_proto)
    if verbose:
        logging.info("Generating %s prototypes as %s" % (num_proto, prototype_policy))
    # Note that we use the original dataset here, not the subsampled one!
    if sub:
        print("Sub prototype")
        prototype = pickle.load(open('prototype.pkl', 'rb')) # to load pre-saved prototype 
    else:
        if prototype_policy == "random":
            if verbose:
                logging.info("Random subset of the initial data.")
            prototype_idx = np.random.permutation(data_original.shape[0])[:num_proto]
            prototype = [data_original[i] for i in prototype_idx]
        elif prototype_policy == "fft":
            prototype_idx = furthest_first_traversal(data_original, num_proto, distance, n_jobs=32)
            # prototype_idx = furthest_first_traversal(data_original, num_proto, distance)
            return prototype_idx
            prototype = [data_original[i] for i in prototype_idx]
            # return prototype
   