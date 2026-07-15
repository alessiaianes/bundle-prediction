"""
matching_1to1.py
----------------
Estende bundle_matching.py con un vincolo 1-a-1:
ogni streamline di A viene accoppiata con una sola streamline di B,
e nessuna streamline di B viene riutilizzata.

Strategia a due livelli per tenere i costi computazionali bassi:
  1. ANN search → riduce il problema da (n_A × n_B) a (n_A × k) candidati
  2. linear_sum_assignment (algoritmo ungherese) solo sulla sottomatrice k candidati
     → O(n_A × k²) invece di O(n³) sul problema completo

Uso consigliato:
  - bundle piccoli/medi (<5000 sl):  match_1to1_full()    — matrice completa, soluzione ottima
  - bundle grandi (>5000 sl):        match_1to1_sparse()  — solo su candidati ANN, molto più veloce
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from dipy.segment.metric import mdf
from functions import hausdorff_mdf, pairwise_mdf
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial import KDTree

# from bundle_matching import (
#     load_bundle,
#     compute_embeddings,
#     build_index_faiss,
#     build_index_kdtree,
#     search_index,
# )


# ---------------------------------------------------------------------------
# Matching 1-a-1 con Hausdorff — matrice completa (bundle piccoli)
# ---------------------------------------------------------------------------

def match_hausdorff_full(path_a: str, path_b: str):
    """
    Matching 1-a-1 usando hausdorff_mdf come matrice di costo completa.

    Soluzione globalmente ottima ma O(n_A × n_B) in memoria e tempo.
    Consigliato per bundle con meno di ~2000-3000 streamlines ciascuno
    (oltre quella soglia la matrice diventa molto grande).

    Ritorna:
      col_ind  : (n_matched,) indici in B
      distances: (n_matched,) distanza hausdorff_mdf del match
      row_ind  : (n_matched,) indici in A
    """
    print(f"Carico bundle A: {path_a}")
    streamlines_a, _ = load_bundle(path_a)
    print(f"  → {len(streamlines_a)} streamlines")

    print(f"Carico bundle B: {path_b}")
    streamlines_b, _ = load_bundle(path_b)
    print(f"  → {len(streamlines_b)} streamlines")

    print("Calcolo matrice Hausdorff completa (n_A × n_B)...")
    cost = hausdorff_mdf(streamlines_a, streamlines_b)

    print("Risolvo assegnamento 1-a-1...")
    row_ind, col_ind = linear_sum_assignment(cost)
    distances = cost[row_ind, col_ind]

    _print_summary(row_ind, col_ind, distances, len(streamlines_a), len(streamlines_b))
    return col_ind, distances, row_ind


# ---------------------------------------------------------------------------
# Matching 1-a-1 con Hausdorff — sparse (bundle grandi)
# ---------------------------------------------------------------------------

def match_hausdorff_sparse(path_a: str, path_b: str, top_k: int = 10):
    """
    Matching 1-a-1 con Hausdorff per bundle grandi.

    Pipeline:
      1. ANN search sull'embedding a 15 float → top_k candidati per ogni A[j]
      2. Calcola hausdorff_mdf SOLO sui candidati (non su tutta la matrice)
      3. linear_sum_assignment sulla matrice sparsa risultante

    Questo riduce il costo da O(n_A × n_B) a O(n_A × top_k),
    mantenendo la qualità geometrica di Hausdorff dove conta.

    Ritorna:
      col_ind  : (n_matched,) indici in B
      distances: (n_matched,) distanza hausdorff_mdf del match
      row_ind  : (n_matched,) indici in A
    """
    print(f"Carico bundle A: {path_a}")
    streamlines_a, _ = load_bundle(path_a)
    n_a = len(streamlines_a)
    print(f"  → {n_a} streamlines")

    print(f"Carico bundle B: {path_b}")
    streamlines_b, _ = load_bundle(path_b)
    n_b = len(streamlines_b)
    print(f"  → {n_b} streamlines")

    # --- Step 1: ANN sull'embedding per trovare i candidati ---
    print("Calcolo embeddings per ANN search...")
    emb_a = compute_embeddings(streamlines_a)
    emb_b = compute_embeddings(streamlines_b)

    index = build_index_faiss(emb_b)
    if index is None:
        print("  FAISS non trovato, uso KD-tree")
        index = build_index_kdtree(emb_b)

    print(f"ANN search (top_k={top_k})...")
    _, candidates = search_index(index, emb_a, k=top_k)
    # candidates: (n_A, top_k) — indici in B

    # --- Step 2: Hausdorff solo sui candidati ---
    # Per ogni streamline j di A, calcola hausdorff_mdf contro i suoi top_k candidati in B.
    # Non costruiamo la matrice completa: per ogni j usiamo solo le top_k streamline di B.
    print("Calcolo Hausdorff sui candidati...")
    BIG = 1e9
    cost = np.full((n_a, n_b), BIG, dtype=np.float32)

    for j in range(n_a):
        cands_j = candidates[j]                          # top_k indici in B
        sl_a_j  = [streamlines_a[j]]                     # lista con una sola streamline
        sl_b_j  = [streamlines_b[i] for i in cands_j]   # lista dei top_k candidati

        # hausdorff su sottoinsiemi di dimensione 1 × top_k
        h = hausdorff_mdf(sl_a_j, sl_b_j)       # shape (1, top_k)

        for ki, i in enumerate(cands_j):
            cost[j, i] = h[0, ki]

    # --- Step 3: assegnamento 1-a-1 ---
    print("Risolvo assegnamento 1-a-1 sulla matrice sparsa...")
    row_ind, col_ind = linear_sum_assignment(cost)

    valid = cost[row_ind, col_ind] < BIG / 2
    row_ind  = row_ind[valid]
    col_ind  = col_ind[valid]
    distances = cost[row_ind, col_ind]

    _print_summary(row_ind, col_ind, distances, n_a, n_b)
    return col_ind, distances, row_ind


# ---------------------------------------------------------------------------
# Utilità
# ---------------------------------------------------------------------------

def _print_summary(row_ind, col_ind, distances, n_a, n_b):
    n_matched = len(row_ind)
    print(f"\n=== Risultati matching 1-a-1 (Hausdorff) ===")
    print(f"  Streamlines in A:   {n_a}")
    print(f"  Streamlines in B:   {n_b}")
    print(f"  Coppie matchate:    {n_matched}  (max possibile: {min(n_a, n_b)})")
    print(f"  Hausdorff medio:    {distances.mean():.3f} mm")
    print(f"  Hausdorff mediana:  {np.median(distances):.3f} mm")
    print(f"  Hausdorff max:      {distances.max():.3f} mm")
    assert len(np.unique(row_ind)) == n_matched, "Violazione vincolo: duplicato in A!"
    assert len(np.unique(col_ind)) == n_matched, "Violazione vincolo: duplicato in B!"
    print("  Vincolo 1-a-1: ✓ verificato")

"""
bundle_matching.py
------------------
Trova per ogni streamline di bundle_A la streamline più simile in bundle_B.

Pipeline:
  1. Carica i .trk e normalizza le streamlines (ricampionamento uniforme)
  2. Calcola un embedding compatto per ogni streamline
  3. Nearest-neighbor search (FAISS se disponibile, altrimenti KD-tree scipy)
  4. Verifica opzionale con MDF sui top-k candidati
  5. Restituisce i match e può salvare il bundle dei matched

Dipendenze:
  pip install dipy nibabel numpy scipy
  pip install faiss-cpu          # opzionale ma consigliato per bundle grandi
"""

import os

import numpy as np
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.tracking.streamline import set_number_of_points
from dipy.segment.metric import mdf


# ---------------------------------------------------------------------------
# 1. I/O
# ---------------------------------------------------------------------------

def load_bundle(trk_path: str):
    """
    Carica un .trk e restituisce lista di streamlines in spazio rasmm.
    """
    sft = load_tractogram(trk_path, "same", bbox_valid_check=False)
    sft.to_rasmm()
    return list(sft.streamlines), sft


def save_bundle(streamlines, reference_sft, out_path: str):
    """
    Salva un sottoinsieme di streamlines come .trk, usando lo stesso
    spazio/affine del tractogram di riferimento.
    """
    out_sft = StatefulTractogram.from_sft(
        streamlines,
        reference_sft,
        # space=Space.RASMM,
    )
    save_tractogram(out_sft, out_path, bbox_valid_check=False)
    print(f"Salvato: {out_path}  ({len(streamlines)} streamlines)")


# ---------------------------------------------------------------------------
# 2. Embedding
# ---------------------------------------------------------------------------

def resample_streamlines(streamlines, n_points: int = 20):
    """
    Ricampiona ogni streamline a n_points equidistanti.
    Necessario per avere vettori di dimensione fissa.
    """
    # for s in streamlines:
    #     set_number_of_points(s, n_points)
    return np.array(set_number_of_points(streamlines, n_points))
    # return [set_number_of_points(s, n_points) for s in streamlines]


def compute_embeddings(streamlines): #, n_points: int = 20) -> np.ndarray:
    """
    Embedding compatto per ogni streamline.

    Strategia a più livelli (tutti concatenati):
      - centroide  (3 float)  → posizione media nello spazio
      - endpoint1  (3 float)  → primo punto
      - endpoint2  (3 float)  → ultimo punto
      - punto medio (3 float) → punto centrale
      - std xyz    (3 float)  → dispersione (proxy di lunghezza/forma)

    Totale: 15 float per streamline — molto leggero, discriminativo in pratica.

    Se vuoi embedding più ricchi (più costosi), imposta use_full_shape=True:
    usa i punti ricampionati appiattiti (n_points*3 float).
    """
    # resampled = resample_streamlines(streamlines, n_points)
    embeddings = []

    for s in streamlines:
        s = np.asarray(s, dtype=np.float32)
        mid = s[len(s) // 2]
        centroid = s.mean(axis=0)
        std = s.std(axis=0)
        vec = np.concatenate([centroid, s[0], s[-1], mid, std])
        embeddings.append(vec)

    return np.array(embeddings, dtype=np.float32)


def compute_embeddings_full_shape(streamlines): #, n_points: int = 20) -> np.ndarray:
    """
    Embedding alternativo: appiattisce i punti ricampionati.
    Più ricco (n_points*3 feature), migliore per bundle con forme simili
    ma orientamenti diversi — ma anche più costoso nell'ANN search.
    Nota: non invariante alla direzione; usa flip_streamlines() prima se necessario.
    """
    # resampled = resample_streamlines(streamlines, n_points)

    return np.array(
        [np.asarray(s, dtype=np.float32).flatten() for s in streamlines],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# 3. Nearest-neighbor search
# ---------------------------------------------------------------------------

def build_index_faiss(embeddings_b: np.ndarray):
    """
    Costruisce un indice FAISS (flat L2) su bundle_B.
    Velocissimo per ricerche su grandi dataset.
    """
    try:
        import faiss
        d = embeddings_b.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings_b)
        return ("faiss", index)
    except ImportError:
        return None


def build_index_kdtree(embeddings_b: np.ndarray):
    """
    Fallback: KD-tree di scipy. Ottimo fino a ~50k streamlines e ~30 feature.
    """
    from scipy.spatial import KDTree
    return ("kdtree", KDTree(embeddings_b))


def search_index(index_tuple, embeddings_a: np.ndarray, k: int = 3):
    """
    Per ogni vettore in embeddings_a trova i k nearest neighbor in index.

    Ritorna:
      distances: (n_a, k) distanze L2
      indices:   (n_a, k) indici in bundle_B
    """
    kind, index = index_tuple
    if kind == "faiss":
        distances, indices = index.search(embeddings_a, k)
    else:  # kdtree
        distances, indices = index.query(embeddings_a, k=k)
        distances = distances ** 2  # converti in L2²  per coerenza con FAISS
    return distances, indices


# ---------------------------------------------------------------------------
# 4. Raffinamento con MDF (opzionale)
# ---------------------------------------------------------------------------

def refine_with_mdf(
    streamlines_a,
    streamlines_b,
    candidate_indices: np.ndarray,
    # n_points: int = 20,
):
    """
    Per ogni streamline in A, sceglie il miglior match tra i candidati
    usando la distanza MDF (Mean of Direct Flip), che è invariante
    alla direzione della streamline.

    candidate_indices: (n_a, k) — output di search_index

    Ritorna:
      best_indices: (n_a,) indice in B del match migliore
      best_mdf:    (n_a,) distanza MDF corrispondente
    """
    # resampled_a = resample_streamlines(streamlines_a, n_points)
    # resampled_b = resample_streamlines(streamlines_b, n_points)

    best_indices = []
    best_distances = []

    for i, candidates in enumerate(candidate_indices):
        sa = np.asarray(streamlines_a[i], dtype=np.float32)
        best_d = np.inf
        best_j = candidates[0]

        for j in candidates:
            sb = np.asarray(streamlines_b[j], dtype=np.float32)
            d = mdf(sa, sb)
            if d < best_d:
                best_d = d
                best_j = j

        best_indices.append(best_j)
        best_distances.append(best_d)

    return np.array(best_indices), np.array(best_distances)


# ---------------------------------------------------------------------------
# 5. Pipeline principale
# ---------------------------------------------------------------------------

def match_bundles(
    path_a: str,
    path_b: str,
    top_k: int = 3,
    use_mdf_refinement: bool = True,
    use_full_shape_embedding: bool = True,
    save_matched_path: str = None,
    mdf_threshold: float = None,
):
    """
    Match completo tra bundle_A e bundle_B.

    Parametri:
      path_a, path_b          : percorsi dei file .trk
      n_points                : punti di ricampionamento per streamline
      top_k                   : candidati ANN su cui applicare MDF
      use_mdf_refinement      : se True, raffina con MDF sui top_k candidati
      use_full_shape_embedding: se True, usa embedding appiattito (n_points*3)
      save_matched_path       : se fornito, salva le streamlines matched di B
      mdf_threshold           : se fornito, esclude match con MDF > soglia (mm)

    Ritorna:
      results: dict con campi
        'match_indices'  : (n_a,) indice in B per ogni streamline di A
        'match_distances': (n_a,) distanza del match (embedding L2² o MDF in mm)
        'streamlines_a'  : lista originale
        'streamlines_b'  : lista originale
    """
    print(f"Carico bundle A: {path_a}")
    streamlines_a, sft_a = load_bundle(path_a)
    print(f"  → {len(streamlines_a)} streamlines")

    print(f"Carico bundle B: {path_b}")
    streamlines_b, sft_b = load_bundle(path_b)
    print(f"  → {len(streamlines_b)} streamlines")

    # --- Embedding ---
    print(f"Calcolo embeddings, full_shape={use_full_shape_embedding})...")
    if use_full_shape_embedding:
        emb_a = compute_embeddings_full_shape(streamlines_a)#, n_points)
        emb_b = compute_embeddings_full_shape(streamlines_b)#, n_points)
    else:
        emb_a = compute_embeddings(streamlines_a)#, n_points)
        emb_b = compute_embeddings(streamlines_b)#, n_points)
    print(f"  dimensione vettore: {emb_a.shape[1]}")

    # --- Indice ANN ---
    print("Costruisco indice ANN...")
    index = build_index_faiss(emb_b)
    if index is None:
        print("  FAISS non trovato, uso KD-tree (scipy)")
        index = build_index_kdtree(emb_b)
    else:
        print("  usando FAISS")

    # --- Search ---
    print(f"ANN search (top_k={top_k})...")
    distances, candidates = search_index(index, emb_a, k=top_k)

    # --- MDF refinement ---
    if use_mdf_refinement and top_k > 1:
        print("Raffinamento MDF sui candidati...")
        match_indices, match_distances = refine_with_mdf(
            streamlines_a, streamlines_b, candidates)#, n_points)

        dist_label = "MDF (mm)"
    else:
        match_indices = candidates[:, 0]
        match_distances = distances[:, 0]
        dist_label = "L2² embedding"

    # --- Statistiche ---
    print(f"\n=== Risultati ===")
    print(f"  Match trovati: {len(match_indices)}")
    print(f"  Distanza media ({dist_label}): {match_distances.mean():.3f}")
    print(f"  Distanza mediana:              {np.median(match_distances):.3f}")
    print(f"  Distanza max:                  {match_distances.max():.3f}")

    # --- Filtro opzionale per soglia MDF ---
    if mdf_threshold is not None and use_mdf_refinement:
        valid = match_distances <= mdf_threshold
        n_valid = valid.sum()
        print(f"  Match sotto soglia {mdf_threshold} mm: {n_valid}/{len(match_indices)}")
    else:
        valid = np.ones(len(match_indices), dtype=bool)

    # --- Salvataggio opzionale ---
    if save_matched_path is not None:
        matched_streamlines = [streamlines_b[i] for i in match_indices[valid]]
        save_bundle(matched_streamlines, sft_b, save_matched_path)

    return {
        "match_indices": match_indices,
        "match_distances": match_distances,
        "valid_mask": valid,
        "streamlines_a": streamlines_a,
        "streamlines_b": streamlines_b,
    }


def save_results(rows, cols, match, folder, filename):
    res = []
    for i in range(len(match)):
        res.append((rows[i], cols[i], match[i]))

    print("Saving results...")
    os.makedirs(folder, exist_ok=True)
    np.save(f"{folder}/{filename}.npy", res)



if __name__ == "__main__":
    from time import perf_counter

    start = perf_counter()
    user = 'alessia.ianes'
    path_sl = f'/home/{user}/Desktop/data/TractoInferno_rearranged/reduced_streamlines'
    path_flip = f'/home/{user}/Desktop/data/TractoInferno_rearranged/flip_bundles'
    # save_trk = f'/home/{user}/Desktop/data/TractoInferno_rearranged/graph_match_trk'
    mdf_threshold = 10.0
    # path_set = f'/home/{user}/Desktop/data/TractoInferno_rearranged/match_euclidean_mdf_threshold/testset'
    for bundle in sorted(os.listdir(path_sl)):
      
        
        if bundle.endswith('_L'):
            bundle_opp = bundle.replace('_L', '_R')
        elif bundle.endswith('_R'):
            bundle_opp = bundle.replace('_R', '_L')
        for set_f in sorted(os.listdir(os.path.join(path_sl, bundle))):
            if set_f != 'testset':
                continue
            for sub in sorted(os.listdir(os.path.join(path_sl, bundle, set_f))):
                if sub not in ['sub-1006', 'sub-1019', 'sub-1024','sub-1046','sub-1047']:
                    continue
                for file in os.listdir(os.path.join(path_sl, bundle, set_f, sub)):
                    


                    path_file = os.path.join(path_sl, bundle, set_f, sub, file)
                    path_opp = [os.path.join(path_flip, bundle_opp, set_f, sub, f) for f in os.listdir(os.path.join(path_flip, bundle_opp, set_f, sub))][0]

                    # saving_path = os.path.join(save_trk, set_f, sub)
                    # os.makedirs(saving_path, exist_ok=True)


                    

                    
                    # # ========== KD TREE + NORMALIZATION ==========

                    # # sl_a, sft_a = load_bundle(path_file)
                    # # sl_b, sft_b = load_bundle(path_opp)

                    # # emb_a = np.array(sft_a.data_per_streamline['embeddings']).astype(np.float32)
                    # # emb_b = np.array(sft_b.data_per_streamline['embeddings']).astype(np.float32)

                    # # print("emb_a shape:", emb_a.shape)
                    # # print("emb_b shape:", emb_b.shape)

                    # # norm_a = emb_a / np.linalg.norm(emb_a, axis=1, keepdims=True)
                    # # norm_b = emb_b / np.linalg.norm(emb_b, axis=1, keepdims=True)
                    # # print("norm_a shape:", norm_a.shape)
                    # # tree = KDTree(norm_b)

                    # # distances, indices = tree.query(norm_a, k=len(norm_b))
                    # # print("distances shape:", distances.shape)
                    # # print("indices shape:", indices.shape)



                    # # row_ind, col_ind = linear_sum_assignment(distances)
                    # # col_ind_oiginal = indices[row_ind, col_ind]

                    # # print("row_ind:", row_ind[:10])
                    # # print("col_ind:", col_ind[:10])

                 


                    # # print("Saving results...")
                    # # match_folder = 'match_KDTree'
                    # # save_results(rwo_ind, col_ind, col_ind_original, f"/home/{user}/Desktop/data/TractoInferno_rearranged/{match_folder}/{set_f}/{sub}", f"{bundle_opp}_to_{bundle}_matched_pairs")


                    # # match_trk = f"/home/{user}/Desktop/data/TractoInferno_rearranged/{match_folder}_trk/{set_f}/{sub}"
                    # # os.makedirs(match_trk, exist_ok=True)
                    # # save_bundle(np.array(sl_b)[col_ind], sft_b, f"{match_trk}/{bundle_opp}_to_{bundle}_32_points.trk")






                    # # ========== COSINE DISTANCE ==========
                    # print(f"Loading bundles {bundle} and flipped {bundle_opp} for sub {sub}...")
                    # sl_a, sft_a = load_bundle(path_file)
                    # sl_b, sft_b = load_bundle(path_opp)

                    # emb_a = np.array(sft_a.data_per_streamline['embeddings']).astype(np.float32)
                    # emb_b = np.array(sft_b.data_per_streamline['embeddings']).astype(np.float32)

                    # print("emb_a shape:", emb_a.shape)
                    # print("emb_b shape:", emb_b.shape)


                    # cos_dist = cosine_distances(emb_a, emb_b)
                    # print("cos_dist shape:", cos_dist.shape)

                    # row_ind, col_ind = linear_sum_assignment(cos_dist)
                    # match_dist = cos_dist[row_ind, col_ind]


                    # print("row_ind:", row_ind[:10])
                    # print("col_ind:", col_ind[:10])
                    # print("distances:", match_dist[:10])

                 
                   


                    # print("Saving results...")
                    # match_folder = 'match_cosine'
                    # # save_results(row_ind, col_ind, match_dist, f"/home/{user}/Desktop/data/TractoInferno_rearranged/{match_folder}/{set_f}/{sub}", f"{bundle_opp}_to_{bundle}_matched_pairs")


                    # # match_trk = f"/home/{user}/Desktop/data/TractoInferno_rearranged/{match_folder}_trk/{set_f}/{sub}"
                    # # os.makedirs(match_trk, exist_ok=True)
                    # # save_bundle(np.array(sl_b)[col_ind], sft_b, f"{match_trk}/{bundle_opp}_to_{bundle}_32_points.trk")

                    
                    # # # ==== TRYING INCORPORATING RESAMPLING + MDF IN COSINE METHOD ====

                    # new_sl_a = resample_streamlines(sl_a, n_points=16)
                    # new_sl_b = resample_streamlines(sl_b, n_points=16)

                    # matched_streamlines_a = np.array(new_sl_a)[row_ind]
                    # matched_streamlines_b = np.array(new_sl_b)[col_ind]

                    # mdf_dist = pairwise_mdf(new_sl_a, new_sl_b)
                    # mdf_diag = np.diag(mdf_dist)

                    
                    # print("Matches:", mdf_diag[:10])


                    # valid = mdf_diag <= 10.0
                    # print(f"Valid matched {valid.sum()}/{len(valid)}")

                    # rows = row_ind[valid]
                    # cols = col_ind[valid]
                    # match_dist = mdf_diag[valid]

                    # print("rows:", rows[:10])
                    # print("cols:", cols[:10])
                    # print("match < 10mm:", match_dist[:10])


                    # print("Distances mm:", match_dist[:10])

                    # save_results(rows, cols, match_dist, f"/home/{user}/Desktop/data/TractoInferno_rearranged/{match_folder}/{set_f}/{sub}", f"{bundle_opp}_to_{bundle}_matched_pairs")




                    















                    # ========== MATCHING 1-1 RESAMPLING TO 16 POINTS ==========
                    print(f"Loading bundles {bundle} and flipped {bundle_opp} for sub {sub}...")
                    sl_a, sft_a = load_bundle(path_file)
                    sl_b, sft_b = load_bundle(path_opp)


                    # emb_a = np.array(sft_a.data_per_streamline['embeddings'])
                    # emb_b = np.array(sft_b.data_per_streamline['embeddings'])

                    # print(emb_a.shape)

                    # print(f"Resampling streamlines to 16 points...")
                    # new_sl_a = resample_streamlines(sl_a, n_points=16)
                    # new_sl_b = resample_streamlines(sl_b, n_points=16)

                    # flip_folder = os.path.join(f'/home/{user}/Desktop/data/TractoInferno_rearranged/flip_bundle_16_points', bundle_opp, set_f, sub)
                    # os.makedirs(flip_folder, exist_ok=True)
                    # reducing_16 = os.path.join(f'/home/{user}/Desktop/data/TractoInferno_rearranged/reduced_streamlines_16_points', bundle, set_f, sub)
                    # os.makedirs(reducing_16, exist_ok=True)
                    # save_bundle(new_sl_a, sft_a, f"{reducing_16}/{bundle}_16_points.trk")
                    # save_bundle(new_sl_b, sft_b, f"{flip_folder}/{bundle_opp}_flipped_16_points.trk")




                    print("Computing distances using pairwise mdf...")
                    # distances = pairwise_mdf(new_sl_a, new_sl_b)
                    distances = pairwise_mdf(np.array(sl_a), np.array(sl_b))
                    # distances = pairwise_mdf(emb_a[:, :, np.newaxis], emb_b[:, :, np.newaxis])

                    # print(distances[:5])



                    print("Checking 1-1 match....")
                    row_ind, col_ind = linear_sum_assignment(distances)
                    matched_pairs = distances[row_ind, col_ind]

                    if mdf_threshold is not None:
                        valid = matched_pairs <= mdf_threshold
                        n_valid = valid.sum()
                        print(f"  Match sotto soglia {mdf_threshold} mm: {n_valid}/{len(col_ind)}")
                    else:
                        valid = np.ones(len(col_ind), dtype=bool)

           

                    match_final = matched_pairs[valid]
                    rows = row_ind[valid]
                    cols = col_ind[valid]
                    res = []

                    for i in range(len(match_final)):
                        res.append((rows[i], cols[i], match_final[i]))

                    len(match_final)
                    print("Saving results...")
                    match_folder = 'match_mdf'
                    save_results(rows, cols, match_final, f"/home/{user}/Desktop/data/TractoInferno_rearranged/{match_folder}/{set_f}/{sub}", f"{bundle_opp}_to_{bundle}_matched_pairs")

                    print(len(sl_a))
                    print(np.array(sl_a).shape)
                    # matched_streamlines = new_sl_b[valid]
                    matched_streamlines = np.array(sl_b)[valid]
                    out_sft = StatefulTractogram.from_sft(
                        np.array(sl_a)[valid],
                        sft_a,
                        # space=Space.RASMM,
                    )
                   
                    out_sl = np.array([np.array(sl) for sl in matched_streamlines])
                    compress = out_sl.reshape(out_sl.shape[0], -1)

                    # # to get original array
                    # # sft.data_per_streamline['match']
                    # # original = sft.reshape[-1, 32, 3]
                  
                    out_sft.data_per_streamline['match'] = compress
                    # out_sft.data_per_streamline['embeddings'] = sft_a.data_per_streamline['embeddings'][valid]  
                    trx_folder = f"/home/{user}/Desktop/data/TractoInferno_rearranged/{match_folder}_trx/{set_f}/{sub}"
                    os.makedirs(trx_folder, exist_ok=True)     
                    save_tractogram(out_sft, f"{trx_folder}/{bundle}_matched.trx", bbox_valid_check=False)

                  

      
                    
                  


                    match_trk = f"/home/{user}/Desktop/data/TractoInferno_rearranged/{match_folder}_trk/{set_f}/{sub}"
                    os.makedirs(match_trk, exist_ok=True)
                    save_bundle(matched_streamlines, sft_b, f"{match_trk}/{bundle_opp}_to_{bundle}_32_points.trk")



            

            

                    



                    

                    

    end = perf_counter()
    print(f"Tempo totale: {end - start:.2f} secondi")



