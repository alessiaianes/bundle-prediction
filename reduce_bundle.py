from time import perf_counter

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from scipy.spatial.distance import cdist

def reduce_bundle_kmeans(streamlines, n_target_streamlines=1000):
    """
    Riduce un fascio a n_target_streamlines scegliendole il più 
    spazialmente distanti e rappresentative possibile.
    """
    if len(streamlines) <= n_target_streamlines:
        return streamlines

    # 1. Calcola l'embedding (i 15 numeri) per tutte le fibre
    embeddings = []
    for s in streamlines:
        s = np.asarray(s, dtype=np.float32)
        mid = s[len(s) // 2]
        centroid = s.mean(axis=0)
        std = s.std(axis=0)
        embeddings.append(np.concatenate([centroid, s[0], s[-1], mid, std]))
    
    embeddings = np.array(embeddings, dtype=np.float32)

    # 2. Clustering ultra-veloce con MiniBatchKMeans
    # MiniBatch è una versione ottimizzata di KMeans per fare molto prima
    kmeans = MiniBatchKMeans(n_clusters=n_target_streamlines, random_state=42, n_init="auto")
    kmeans.fit(embeddings)
    
    # 3. Per ogni centroide (gruppo), trova la fibra reale più vicina
    # Calcola la distanza tra tutti gli embeddings e i centroidi del K-Means
    distances = cdist(kmeans.cluster_centers_, embeddings, metric='euclidean')
    
    # Prendi l'indice della fibra con la distanza minima per ogni cluster
    representative_indices = np.argmin(distances, axis=1)
    
    # Estrai le fibre finali (e assicurati che non ci siano duplicati)
    unique_indices = np.unique(representative_indices)
    reduced_streamlines = [streamlines[i] for i in unique_indices]

    return reduced_streamlines


import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist

def reduce_bundle_agglomerative(streamlines, n_target_streamlines=1000):
    """
    Riduce un fascio usando Agglomerative Clustering.
    Attenzione: molto lento se len(streamlines) > 15.000.
    """
    if len(streamlines) <= n_target_streamlines:
        return streamlines

    # 1. Calcola l'embedding (i 15 numeri) per tutte le fibre
    embeddings = []
    for s in streamlines:
        s = np.asarray(s, dtype=np.float32)
        mid = s[len(s) // 2]
        centroid = s.mean(axis=0)
        std = s.std(axis=0)
        embeddings.append(np.concatenate([centroid, s[0], s[-1], mid, std]))
    
    embeddings = np.array(embeddings, dtype=np.float32)

    # 2. Clustering Agglomerativo
    # 'ward' è il linkage migliore perché minimizza la varianza nei gruppi
    agglo = AgglomerativeClustering(n_clusters=n_target_streamlines, linkage='ward')
    labels = agglo.fit_predict(embeddings)
    
    # 3. Trova la fibra rappresentativa per ogni cluster
    representative_indices = []
    
    for cluster_id in range(n_target_streamlines):
        # Prendi gli indici di tutte le fibre che sono finite in questo cluster
        points_in_cluster_idx = np.where(labels == cluster_id)[0]
        
        # Se il cluster ha una sola fibra, prendi quella
        if len(points_in_cluster_idx) == 1:
            representative_indices.append(points_in_cluster_idx[0])
            continue
            
        # Estrai i vettori di queste fibre
        cluster_embeddings = embeddings[points_in_cluster_idx]
        
        # Calcola il centroide "virtuale" facendone la media
        cluster_center = cluster_embeddings.mean(axis=0, keepdims=True)
        
        # Trova quale fibra del cluster è più vicina al centroide
        dists = cdist(cluster_center, cluster_embeddings, metric='euclidean')
        local_best_idx = np.argmin(dists)
        
        # Risali all'indice originale della fibra e salvalo
        global_best_idx = points_in_cluster_idx[local_best_idx]
        representative_indices.append(global_best_idx)

    # 4. Estrai le fibre finali
    reduced_streamlines = [streamlines[i] for i in representative_indices]

    return reduced_streamlines



import numpy as np

def reduce_bundle_hybrid_dynamic(streamlines_a, streamlines_b, reduce_a=True):
    """
    Riduce il fascio applicando il vincolo dinamico:
    target = min(len(A), len(B), 5000)
    
    Se reduce_a=True riduce il fascio A, altrimenti riduce il fascio B.
    """
    len_a = len(streamlines_a)
    len_b = len(streamlines_b)
    
    # 1. Calcolo del target dinamico secondo la tua regola
    target_final = min(len_a, len_b, 5000)
    
    # Seleziona quale dei due fasci stiamo riducendo in questa chiamata
    current_streamlines = streamlines_a if reduce_a else streamlines_b
    total_current = len(current_streamlines)
    
    print(f"\n--- Riduzione Fascio (Iniziali: {total_current}) ---")
    print(f"  Target finale calcolato: {target_final}")
    
    # Se il fascio ha già meno fibre del target, non c'è nulla da ridurre
    if total_current <= target_final:
        print("  Il fascio ha già un numero di fibre inferiore o uguale al target. Salto riduzione.")
        return current_streamlines

    # 2. Gestione della pipeline in base alla mole di dati
    # Impostiamo un target intermedio per il K-Means (es. 15.000 fibre)
    target_intermediate = 10000
    
    if total_current > target_intermediate:
        # Dataset enorme (centinaia di migliaia): serve la fase 1 col K-Means
        print(f"  [Fase 1] Dataset enorme. K-Means veloce: {total_current} -> {target_intermediate} fibre")
        intermediate_streamlines = reduce_bundle_kmeans(current_streamlines, n_target_streamlines=target_intermediate)
        
        print(f"  [Fase 2] Agglomerativo di precisione: {target_intermediate} -> {target_final} fibre")
        final_streamlines = reduce_bundle_agglomerative(intermediate_streamlines, n_target_streamlines=target_final)
    else:
        # Dataset medio (es. 12.000 fibre): saltiamo il K-Means e andiamo diretti di Agglomerativo
        print(f"  [Fase Unica] Dataset gestibile. Agglomerativo diretto: {total_current} -> {target_final} fibre")
        final_streamlines = reduce_bundle_agglomerative(current_streamlines, n_target_streamlines=target_final)
        
    return final_streamlines

def save_bundle(streamlines, reference_sft, out_path: str):
    """
    Salva un set di streamlines come .trk, ereditando lo spazio e l'affine
    dal tractogramma di riferimento originale.
    """
    out_sft = StatefulTractogram.from_sft(
        streamlines,
        reference_sft,
        # space=Space.RASMM,
    )
    save_tractogram(out_sft, out_path, bbox_valid_check=False)
    print(f"  [Output] Salvato .trk ridotto: {out_path} ({len(streamlines)} streamlines)")


from matching import load_bundle
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.io.streamline import save_tractogram
import os



def reduce(user, path, red_path):
    start = perf_counter()

    # user = 'alessia.ianes'
    # path = f'/home/{user}/Desktop/data/TractoInferno_rearranged/bundles'
    # red_path = f'/home/{user}/Desktop/data/TractoInferno_rearranged/reduced_bundles_dynamic'
    


    for bundle in sorted(os.listdir(path)):
        if 'AF' not in bundle:
            continue
        bundle_path = os.path.join(path, bundle)
        red_bundle_path_a = os.path.join(red_path, bundle)
        red_bundle_path_b = os.path.join(red_path, bundle.replace('_L', '_R') if '_L' in bundle else bundle.replace('_R', '_L'))
        os.makedirs(red_bundle_path_a, exist_ok=True)
        os.makedirs(red_bundle_path_b, exist_ok=True)

        opp_path = os.path.join(path, bundle.replace('_L', '_R') if '_L' in bundle else bundle.replace('_R', '_L'))

        for set_f in sorted(os.listdir(bundle_path)):
            if set_f != 'testset':
                continue
            set_path = os.path.join(bundle_path, set_f)

            for sub in sorted(os.listdir(set_path)):
                sub_path = os.path.join(set_path, sub)

                for file in os.listdir(sub_path):
                    if not file.endswith('.trk') or os.path.exists(os.path.join(red_bundle_path_a, set_f, sub, f'{sub}__{bundle}_reduced.trk')):
                        continue
                    
                    path_file = os.path.join(sub_path, file)

                    # opp_name = file.replace('_32_points.trk', '_opp_32_points.trk')
                    path_opp = [os.path.join(opp_path, set_f, sub, f) for f in os.listdir(os.path.join(opp_path, set_f, sub))][0]

                    if not os.path.exists(path_opp):
                        print(f"Opposite bundle not found for {file}. Skipping.")
                        continue

                    print(f"\nProcessing {file} and its opposite {bundle.replace('_L', '_R') if '_L' in bundle else bundle.replace('_R', '_L')}...")
                    
                    # Carichi i due fasci originali
                    streamlines_a, sft_a = load_bundle(path_file)
                    streamlines_b, sft_b = load_bundle(path_opp)

                    # Riduci entrambi i fasci allo stesso identico numero di streamlines
                    streamlines_a_reduced = reduce_bundle_hybrid_dynamic(streamlines_a, streamlines_b, reduce_a=True)
                    streamlines_b_reduced = reduce_bundle_hybrid_dynamic(streamlines_a, streamlines_b, reduce_a=False)

                    out_folder_a = os.path.join(red_bundle_path_a, set_f, sub)
                    out_folder_b = os.path.join(red_bundle_path_b, set_f, sub)

                    os.makedirs(out_folder_a, exist_ok=True)
                    os.makedirs(out_folder_b, exist_ok=True)

                    # Definiamo i nomi dei nuovi file trk
                    name_reduced_a = f'{sub}__{bundle}_reduced.trk'
                    name_reduced_b = f"{sub}__{bundle.replace('_L', '_R') if bundle.endswith('_L') else bundle.replace('_R', '_L')}_reduced.trk"

                    
                    path_reduced_a = os.path.join(out_folder_a, name_reduced_a)
                    path_reduced_b = os.path.join(out_folder_b, name_reduced_b)

                    # Salviamo i file ereditando le informazioni geometriche corrette (sft_a e sft_b)
                    save_bundle(streamlines_a_reduced, sft_a, path_reduced_a)
                    save_bundle(streamlines_b_reduced, sft_b, path_reduced_b)
            
            end = perf_counter()
            print(f"Time: {end - start}")
