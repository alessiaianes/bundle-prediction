# IMPLEMENTATION FROM https://github.com/FBK-NILab/tractome/blob/master/tractome/compute.py

import json
import logging
import tempfile

from dipy.utils.optpkg import optional_package
import numpy as np
from scipy.ndimage import affine_transform
from sklearn.cluster import MiniBatchKMeans
import wgpu

from fury import actor, window
import pickle
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram

ray, has_ray, _ = optional_package("ray")


# def furthest_first_traversal(S, k, distance, permutation=True):
#     print(len(S))
#     print(k)
#     print(distance)
#     """This is the farthest first traversal (fft) algorithm which is
#     known to be a good sub-optimal solution to the k-center problem.

#     See for example:
#     Hochbaum, Dorit S. and Shmoys, David B., A Best Possible Heuristic
#     for the k-Center Problem, Mathematics of Operations Research, 1985.

#     or: http://en.wikipedia.org/wiki/Metric_k-center
#     """
#     do an initial permutation of S, just to be sure that objects in
#     S have no special order. Note that this permutation does not
#     affect the original S.
#     if permutation:
#         print("In permutation")
#         idx = np.random.permutation(S.shape[0])
#         S = S[idx]
#     else:
#         idx = np.arange(S.shape[0], dtype=np.int32)
#     T = [0]
#     while len(T) < k:
#         z = distance(S, S[T]).min(1).argmax()
#         T.append(z)
#         print(len(T), end='\r')
  
   
#     return idx[T]


import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial


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
            print("I'm not back")
            # pickle.dump(prototype, open('prototype.pkl', 'wb'))
            

        elif prototype_policy == "sff":
            prototype_idx = subset_furthest_first(data_original, num_proto, distance)
            prototype = [data_original[i] for i in prototype_idx]
        else:
            raise Exception("Unknown prototype policy: %s" % prototype_policy)
    
    # prototype = pickle.load(open('prototype.pkl', 'rb')) # to load pre-saved prototype 
    

    if verbose:
        logging.info("Computing dissimilarity matrix.")
    if has_ray and n_jobs > 1:
        logging.info(
            "Parallel computation of the dissimilarity matrix: %s cpus." % n_jobs
        )

        tmp = np.linspace(0, data.shape[0], n_jobs).astype(np.int32)
        chunks = zip(tmp[:-1], tmp[1:])

        tmp_dir = tempfile.TemporaryDirectory()

        if not ray.is_initialized():
            ray.init(
                _system_config={
                    "object_spilling_config": json.dumps(
                        {
                            "type": "filesystem",
                            "params": {"directory_path": tmp_dir.name},
                        }
                    )
                }
            )

        func = ray.remote(distance)
        func_refs = [func.remote(data[start:end], prototype) for start, end in chunks]

        data_dissimilarity = []
        for i in range(len(func_refs)):
            data_dissimilarity.extend(ray.get(func_refs[i]))

    else:
       
        data_dissimilarity = distance(data_original, prototype)
     

    return data_dissimilarity


def mkbm_clustering(dissimilarity_matrix, n_clusters, streamline_ids):
    """Perform MKBM clustering on the dissimilarity matrix.

    Parameters
    ----------
    dissimilarity_matrix : ndarray
        The dissimilarity matrix to cluster.
    n_clusters : int
        The number of clusters to create.
    streamline_ids : ndarray
        The IDs of the streamlines to cluster.

    Returns
    -------
    dict
        A dictionary mapping cluster centers to lists of streamline IDs.
    """
    streamline_ids = np.asarray(streamline_ids, dtype=np.int32)
    dissimilarity_matrix = dissimilarity_matrix[streamline_ids]

    logging.info(f"Clustering with MKBM with {n_clusters} clusters")
    mbkm = MiniBatchKMeans(
        init="random",
        n_clusters=n_clusters,
        batch_size=1000,
        n_init=10,
        max_no_improvement=5,
        verbose=0,
    )
    mbkm.fit(dissimilarity_matrix)

    medoids_exhs = np.zeros(n_clusters, dtype=np.int32)
    idxs = []
    for i, centroid in enumerate(mbkm.cluster_centers_):
        idx_i = np.where(mbkm.labels_ == i)[0]
        if idx_i.size == 0:
            idx_i = [0]
        tmp = dissimilarity_matrix[idx_i] - centroid
        medoids_exhs[i] = streamline_ids[idx_i[(tmp * tmp).sum(1).argmin()]]
        idxs.append(streamline_ids[idx_i].tolist())

    clusters = dict(zip(medoids_exhs, idxs))
    return clusters


def calculate_filter(rois, *, flip=None, reference_shape=None):
    """Calculate a combined ROI filter using logical AND.

    Parameters
    ----------
    rois : ndarray
        ROI volumes to combine with shape (X, Y, Z).
    flip : Sequence[bool] or None, optional
        Per-ROI flag indicating whether the ROI should be inverted
        before combination. If None, all ROIs are inverted.
    reference_shape : tuple[int, ...] or None, optional
        Expected ROI shape. If None, shape from the first ROI is used.

    Returns
    -------
    ndarray
        Boolean mask resulting from a logical AND across all ROIs
        (after optional inversion).

    Raises
    ------
    ValueError
        If no ROIs are provided, if `flip` length does not match
        `rois`, or if no ROI matches `reference_shape`.
    """
    if rois is None or len(rois) == 0:
        raise ValueError("At least one ROI must be provided.")

    if flip is None:
        flip = [False] * len(rois)

    if len(flip) != len(rois):
        raise ValueError(
            "The `flip` list must have the same length as `rois` "
            f"({len(flip)} != {len(rois)})."
        )

    if reference_shape is None:
        reference_shape = np.asarray(rois[0]).shape
    else:
        reference_shape = tuple(reference_shape)
    combined_mask = np.ones(reference_shape, dtype=bool)
    matched_count = 0

    for idx, (roi, should_flip) in enumerate(zip(rois, flip)):
        roi_mask = np.asarray(roi).astype(bool, copy=False)

        if roi_mask.shape != reference_shape:
            logging.warning(
                "Skipping ROI %s due to shape mismatch: expected %s, got %s.",
                idx,
                reference_shape,
                roi_mask.shape,
            )
            continue

        if bool(should_flip):
            roi_mask = np.logical_not(roi_mask)

        combined_mask = np.logical_and(combined_mask, roi_mask)
        matched_count += 1

    if matched_count == 0:
        raise ValueError(
            f"No ROI matched the reference shape. Expected shape: {reference_shape}."
        )

    return combined_mask


def create_roi_from_world(bounds, affine, center, radius, *, type="spherical"):
    """Create a binary spherical ROI from world-space center and radius.

    Parameters
    ----------
    bounds : tuple[int, int, int]
        ROI output shape in voxel coordinates.
    affine : ndarray, shape (4, 4)
        Voxel-to-world affine transform.
    center : Sequence[float]
        Sphere center in world coordinates.
    radius : float
        Sphere radius in world units.
    type : str, optional
        Type of ROI to create. Currently only "spherical" is supported.

    Returns
    -------
    tuple[ndarray, ndarray]
        `(roi, affine)` where `roi` is a uint8 binary mask with ones inside
        the sphere and zeros elsewhere.
    """
    bounds = tuple(int(v) for v in bounds)
    if len(bounds) != 3:
        raise ValueError(f"`bounds` must have 3 dimensions, got {bounds}.")

    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError(f"`affine` must have shape (4, 4), got {affine.shape}.")

    center = np.asarray(center, dtype=np.float64)
    if center.shape != (3,):
        raise ValueError(f"`center` must have shape (3,), got {center.shape}.")
    if radius < 0:
        raise ValueError("`radius` must be non-negative.")

    roi = np.zeros(bounds, dtype=np.uint8)

    inv_affine = np.linalg.inv(affine)
    center_vox = (inv_affine @ np.r_[center, 1.0])[:3]
    inv_linear = inv_affine[:3, :3]
    voxel_radii = np.linalg.norm(inv_linear * float(radius), axis=0)
    radius_vox = float(np.mean(voxel_radii))

    grid = np.indices(bounds, dtype=np.float32)
    dist_sq = (
        (grid[0] - center_vox[0]) ** 2
        + (grid[1] - center_vox[1]) ** 2
        + (grid[2] - center_vox[2]) ** 2
    )
    roi[dist_sq <= (radius_vox**2)] = 1

    return roi, affine


def transform_roi_to_world_grid(roi_data, affine, *, cval=0.0, threshold=0.5):
    """Resample ROI data to an axis-aligned world-coordinate grid.

    The returned array is indexed in world space using `world_min` as origin:
    `world_index = world_coord - world_min`.

    Parameters
    ----------
    roi_data : ndarray
        Input ROI volume in voxel coordinates.
    affine : ndarray, shape (4, 4)
        Voxel-to-world affine transform.
    cval : float, optional
        Constant value for out-of-bounds sampling.
    threshold : float or None, optional
        If not None, output is binarized with ``>= threshold``.

    Returns
    -------
    tuple[ndarray, ndarray]
        `(transformed_data, world_min)` where:
        - `transformed_data` is the ROI in world-grid indexing.
        - `world_min` is the minimum world coordinate (x, y, z) used as origin.

    Raises
    ------
    ValueError
        If `roi_data` is not 3D or `affine` is not 4x4.
    """
    roi_data = np.asarray(roi_data)
    if roi_data.ndim != 3:
        raise ValueError(f"`roi_data` must be a 3D array, got shape {roi_data.shape}.")

    affine = np.asarray(affine, dtype=np.float64)
    if affine.shape != (4, 4):
        raise ValueError(f"`affine` must have shape (4, 4), got {affine.shape}.")

    if np.linalg.det(affine[:3, :3]) == 0:
        raise ValueError("`affine` is singular and cannot be inverted.")

    max_idx = np.asarray(roi_data.shape, dtype=np.float64) - 1.0
    corners_ijk = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [max_idx[0], 0.0, 0.0],
            [0.0, max_idx[1], 0.0],
            [0.0, 0.0, max_idx[2]],
            [max_idx[0], max_idx[1], 0.0],
            [max_idx[0], 0.0, max_idx[2]],
            [0.0, max_idx[1], max_idx[2]],
            [max_idx[0], max_idx[1], max_idx[2]],
        ],
        dtype=np.float64,
    )
    corners_h = np.c_[corners_ijk, np.ones(len(corners_ijk), dtype=np.float64)]
    world_corners = (affine @ corners_h.T).T[:, :3]

    world_min = np.floor(world_corners.min(axis=0)).astype(np.int32)
    world_max = np.ceil(world_corners.max(axis=0)).astype(np.int32)
    output_shape = tuple((world_max - world_min + 1).astype(np.int32))

    inv_affine = np.linalg.inv(affine)
    matrix = inv_affine[:3, :3]
    offset = (inv_affine[:3, :3] @ world_min) + inv_affine[:3, 3]

    transformed_data = affine_transform(
        roi_data.astype(np.float32, copy=False),
        matrix=matrix,
        offset=offset,
        output_shape=output_shape,
        order=1,
        mode="constant",
        cval=float(cval),
        prefilter=False,
    )

    if threshold is not None:
        transformed_data = transformed_data >= threshold

    return transformed_data, world_min


def _fetch_positions_from_gpu(show_manager, geom_positions_buffer, *, sync_cpu=False):
    """Read back geometry.positions from GPU into a NumPy array.

    Notes
    -----
    This uses pygfx/wgpu internals (`_wgpu_object`) and requires COPY_SRC usage.
    """
    wgpu_buffer = getattr(geom_positions_buffer, "_wgpu_object", None)
    if wgpu_buffer is None:
        return None

    raw = show_manager.device.queue.read_buffer(wgpu_buffer)
    cpu_shape = np.asarray(geom_positions_buffer.data).shape
    gpu_positions = np.frombuffer(raw, dtype=np.float32).reshape(cpu_shape).copy()

    if sync_cpu and geom_positions_buffer.data is not None:
        np.asarray(geom_positions_buffer.data)[...] = gpu_positions

    return gpu_positions


def _get_line_ids_from_positions(wobj, positions):
    """Return kept/filtered original line ids from a flat positions buffer."""
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    offsets = np.asarray(wobj._line_offsets, dtype=np.int64)
    lengths = np.asarray(wobj._line_lengths, dtype=np.int64)

    kept_ids = []
    filtered_ids = []
    for line_id, (offset, length) in enumerate(zip(offsets, lengths)):
        segment = positions[offset : offset + length]
        if np.isfinite(segment).all():
            kept_ids.append(line_id)
        else:
            filtered_ids.append(line_id)

    return kept_ids, filtered_ids


def filter_streamline_ids(streamlines, roi, *, origin=(0, 0, 0)):

    # TODO: Remove after FURY v2.0.0a6
    if roi is not None and roi.ndim == 3:
        roi = np.swapaxes(roi, 0, 2)

    max = np.asarray(streamlines[0], dtype=np.float32).max(axis=0)
    min = np.asarray(streamlines[0], dtype=np.float32).min(axis=0)

    scene = window.Scene()
    filtered_streamlines = actor.streamlines(
        streamlines, roi_mask=roi, roi_origin=origin
    )
    points = actor.point(
        np.asarray([min, max], dtype=np.float32),
        colors=np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32),
    )
    filtered_streamlines.geometry.positions._wgpu_usage |= wgpu.BufferUsage.COPY_SRC
    offscreen_showm = window.ShowManager(scene=scene, window_type="offscreen")
    scene.add(points)
    scene.add(filtered_streamlines)
    offscreen_showm.render()
    offscreen_showm.window.draw()
    filtered_positions = _fetch_positions_from_gpu(
        offscreen_showm, filtered_streamlines.geometry.positions
    )
    filtered_positions = np.asarray(filtered_positions, dtype=np.float32).reshape(-1, 3)
    kept_ids, _ = _get_line_ids_from_positions(filtered_streamlines, filtered_positions)
    offscreen_showm.close()
    return kept_ids
