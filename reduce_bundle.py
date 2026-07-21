import os
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from dipy.io.streamline import save_tractogram
from dipy.tracking.streamline import set_number_of_points
from dipy.io.stateful_tractogram import StatefulTractogram
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering


from matching import load_bundle



def reduce_bundle_kmeans(streamlines, n_target_streamlines=1000):
    """
    Reduce a bundle to n_target_streamlines choosing them
    as far and representative as possible.
    """
    if len(streamlines) <= n_target_streamlines:
        return streamlines
    

    sl_res = set_number_of_points(streamlines, 32)

    matrix = np.array([sl.flatten() for sl in sl_res], dtype=np.float32)

    # 1. "Fast" clustering with MiniBatchKMeans
    kmeans = MiniBatchKMeans(n_clusters=n_target_streamlines, random_state=42, n_init="auto")
    kmeans.fit(matrix)
    
    # 2. For each centroid, look for the nearest real streamline.
    # Compute distance among every streamline and centroid of K-Means
    distances = cdist(kmeans.cluster_centers_, matrix, metric='euclidean')
    
    # 3. Take the index of the streamline with the minimum distance for each cluster
    representative_indices = np.argmin(distances, axis=1)
    
    # 4. Extract final streamlines
    unique_indices = np.unique(representative_indices)
    reduced_streamlines = [streamlines[i] for i in unique_indices]

    return reduced_streamlines




def reduce_bundle_agglomerative(streamlines, n_target_streamlines=1000):
    """
    Reduce a bundle using Agglomerative Clustering.
    """
    if len(streamlines) <= n_target_streamlines:
        return streamlines


    sl_res = set_number_of_points(streamlines, 32)
    matrix = np.array([sl.flatten() for sl in sl_res], dtype=np.float32)


    # 1. Initializing clustering
    
    agglo = AgglomerativeClustering(n_clusters=n_target_streamlines, linkage='ward')
    labels = agglo.fit_predict(matrix)


    
    dist_matrix = pairwise_distances(matrix, metric='euclidean', n_jobs=32)

    print("Matrix computed! Starting clustering...")


    # 2. Initialize agglomerative clustering given the matrix already computed
    agg_clustering = AgglomerativeClustering(
        n_clusters=n_target_streamlines,
        metric='precomputed',
        linkage='average' 
    )

    labels = agg_clustering.fit_predict(dist_matrix)




    # 2. Look for the most representative streamline for each cluster
    representative_indices = []
    
    for cluster_id in range(n_target_streamlines):
        # Take the indexes of all streamlines of clusetr "cluster_id"
        points_in_cluster_idx = np.where(labels == cluster_id)[0]
        
       
        if len(points_in_cluster_idx) == 1:
            representative_indices.append(points_in_cluster_idx[0])
            continue
            
        # Extract streamlines for those indexes
        cluster_embeddings = matrix[points_in_cluster_idx]
        
        # Compute "virtual" centroid (mean)
        cluster_center = cluster_embeddings.mean(axis=0, keepdims=True)
        
        # Look for the nearest streamline to the centroid
        dists = cdist(cluster_center, cluster_embeddings, metric='euclidean')
        local_best_idx = np.argmin(dists)
        
        # Save the streamlines indexes
        global_best_idx = points_in_cluster_idx[local_best_idx]
        representative_indices.append(global_best_idx)

    # 3. Extract the streamlines
    reduced_streamlines = [streamlines[i] for i in representative_indices]

    return reduced_streamlines



def reduce_bundle_hybrid_dynamic(streamlines_a, streamlines_b, target_number, reduce_a=True):
    """
    Reduce bundle apllying dynamic constraint:
    target = min(len(A), len(B), 10000)
    
    If reduce_a=True bundle A is reduced, otherwise bundle B.
    """
    len_a = len(streamlines_a)
    len_b = len(streamlines_b)
    
    # 1. Compute how many streamlines to keep (the minimum among 10000, the number of streamlines of A and the number of streamlines of B)
    target_final = min(len_a, len_b, target_number)
    
    # Select which bundle to reduce
    current_streamlines = streamlines_a if reduce_a else streamlines_b

    current_streamlines = streamlines_a if reduce_a else streamlines_b
    total_current = len(current_streamlines)
    
    print(f"\n--- Reducing bundle (Initial number of streamlines: {total_current}) ---")
    print(f"  Streamlines to look for: {target_final}")
    
  
    if total_current <= target_final:
        print("  The bundle has a number of streamlines already under target. Therefore we skip reduction.")
        return current_streamlines

    # 2. Manage the pipeline
    # We fix a target to reduce the number of streamlines from original to k-means, so that agglomerative clustering will take less
    target_intermediate = target_number + (target_number // 2)
    
    if total_current > target_intermediate:
        print(f"  [Phase 1] K-Means: {total_current} -> {target_intermediate} streamlines")
        intermediate_streamlines = reduce_bundle_kmeans(current_streamlines, n_target_streamlines=target_intermediate)
        
        print(f"  [Phase 2] Agglomerative clustering: {target_intermediate} -> {target_final} streamlines")
        final_streamlines = reduce_bundle_agglomerative(intermediate_streamlines, n_target_streamlines=target_final)
    else:
        print(f"  [Single phase] Agglomerative: {total_current} -> {target_final} streamlines")
        final_streamlines = reduce_bundle_agglomerative(current_streamlines, n_target_streamlines=target_final)
        
    return final_streamlines

def save_bundle(streamlines, reference_sft, out_path: str):
    """
    Save a set of streamlines ad .trk, taking originale affine.
    """
    out_sft = StatefulTractogram.from_sft(
        streamlines,
        reference_sft
    )
    save_tractogram(out_sft, out_path, bbox_valid_check=False)
    print(f"  [Output] Reduced bundle saved .trk: {out_path} ({len(streamlines)} streamlines)")






def process_subject(sub, set_path, bundle, bundle_path, opp_path, red_bundle_path_a, red_bundle_path_b, set_f, target_number):
    """Processa un singolo soggetto per una coppia di bundle."""
    sub_path = os.path.join(set_path, sub)

    for file in os.listdir(sub_path):
        if not file.endswith('.trk'):
            continue

        out_file_a = os.path.join(red_bundle_path_a, set_f, sub, f'{sub}__{bundle}_reduced.trk')
        if os.path.exists(out_file_a):
            continue

        path_file = os.path.join(sub_path, file)

        opp_sub_dir = os.path.join(opp_path, set_f, sub)
        opp_files = os.listdir(opp_sub_dir)
        if not opp_files:
            print(f"Opposite bundle not found for {file}. Skipping.")
            continue

        path_opp = os.path.join(opp_sub_dir, opp_files[0])

        if not os.path.exists(path_opp):
            print(f"Opposite bundle not found for {file}. Skipping.")
            continue

        opp_bundle = bundle.replace('_L', '_R') if '_L' in bundle else bundle.replace('_R', '_L')
        print(f"\nProcessing {file} and its opposite {opp_bundle}...")


        # Load bundles
        streamlines_a, sft_a = load_bundle(path_file)
        streamlines_b, sft_b = load_bundle(path_opp)

        # Reduce number of streamlines
        streamlines_a_reduced = reduce_bundle_hybrid_dynamic(streamlines_a, streamlines_b, target_number, reduce_a=True)
        streamlines_b_reduced = reduce_bundle_hybrid_dynamic(streamlines_a, streamlines_b, target_number, reduce_a=False)

        out_folder_a = os.path.join(red_bundle_path_a, set_f, sub)
        out_folder_b = os.path.join(red_bundle_path_b, set_f, sub)
        os.makedirs(out_folder_a, exist_ok=True)
        os.makedirs(out_folder_b, exist_ok=True)

        name_reduced_a = f'{sub}__{bundle}_reduced.trk'
        name_reduced_b = f'{sub}__{opp_bundle}_reduced.trk'

        save_bundle(streamlines_a_reduced, sft_a, os.path.join(out_folder_a, name_reduced_a))
        save_bundle(streamlines_b_reduced, sft_b, os.path.join(out_folder_b, name_reduced_b))

    return sub  # useful for logging



def bundle_reduction(path, set_check, red_path, target_number, max_workers=4):

    for bundle in sorted(os.listdir(path)):
        # if 'AF' not in bundle:
        #     continue

        bundle_path = os.path.join(path, bundle)
        opp_bundle = bundle.replace('_L', '_R') if '_L' in bundle else bundle.replace('_R', '_L')
        red_bundle_path_a = os.path.join(red_path, bundle)
        red_bundle_path_b = os.path.join(red_path, opp_bundle)
        os.makedirs(red_bundle_path_a, exist_ok=True)
        os.makedirs(red_bundle_path_b, exist_ok=True)

        opp_path = os.path.join(path, opp_bundle)

        for set_f in sorted(os.listdir(bundle_path)):
            if set_f != set_check:
                continue

            set_path = os.path.join(bundle_path, set_f)
            subjects = sorted(os.listdir(set_path))

            # Parallelize on subjects
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        process_subject,
                        sub, set_path, bundle, bundle_path,
                        opp_path, red_bundle_path_a, red_bundle_path_b, set_f, target_number
                    ): sub
                    for sub in subjects
                }

                for future in tqdm(as_completed(futures), total=len(futures)):
                    sub = futures[future]
                    try:
                        future.result()
                        print(f"✓ Sub {sub} completed")
                    except Exception as e:
                        print(f"✗ Error for sub {sub}: {e}")
          