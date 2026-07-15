import os
import subprocess

from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.distance import euclidean

from tqdm import tqdm
import numpy as np
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.tracking.utils import density_map
from sklearn.cluster import AgglomerativeClustering
import csv
import nibabel as nib
from compute import compute_dissimilarity
from functions import *
from time import perf_counter

def ants_registration(fixed, moving, out_path, transform):

    """

        :param fixed: Target anatomy space
        :param moving: Moving anatomy
        :param out_path: Out path for the result
        :param transform: Transform to use. Valid options are 'rigid', 'affine', 'SyN'
        :return: the output path

        Calls as a subprocess the script run_ANTS.sh from the repo at https://github.com/FBK-NILab/tractogram_alignment

    """

    # os.makedirs(out_path, exist_ok=True)
    cmd = ['bash', '/home/alessia.ianes/Desktop/tractogram_alignment/code/wm_registration/run_ANTs.sh', moving, fixed, out_path, transform]
    subprocess.check_call(cmd , stdout=subprocess.DEVNULL)


def scil_apply_transform_to_trk(trk, t1, mat, out, deformation=None):

    # os.makedirs(out, exist_ok=True)

    trk_name = trk.split('/')[-1]
    out = os.path.join(out, trk_name.replace('.trk', '_aligned_MNI.trk').replace('.tck', '_aligned_MNI.trk'))

    cmd = ['scil_tractogram_apply_transform', '--inverse', '--remove_invalid', '-f', '--reference', t1, trk, t1, mat, out]

    if deformation is not None:
        cmd.extend(['--in-deformation', deformation])

    subprocess.check_call(cmd, stdout=subprocess.DEVNULL)


def process_sub(data_path, out_path, ref_path, set_path, s):

    anatomy_path = os.path.join(data_path, 'anat', set_path, s)
    out_anatomy_path = os.path.join(out_path, 'anat', set_path, s)

    bundles_path = os.path.join(data_path, 'bundles', set_path, s)
    out_bundles_path = os.path.join(out_path, 'bundles', set_path, s)


    os.makedirs(out_anatomy_path, exist_ok=True)
    os.makedirs(out_bundles_path, exist_ok=True)


    # sub_path = os.path.join(data_path, s)

    # out_path_sub = os.path.join(out_path, s)
    # os.makedirs(out_path_sub, exist_ok=True)

    # sub_anatomy = os.path.join(sub_path, 'anat')
    anatomy_file = os.listdir(anatomy_path)[0]
    anatomy_file = os.path.join(anatomy_path, anatomy_file)

    # sub_tractograms = os.path.join(sub_path, 'tractograms')
    # sub_out_tractograms = os.path.join(out_path_sub, 'tractograms')
    trk_files = os.listdir(bundles_path)
    trk_files = [os.path.join(bundles_path, item) for item in trk_files]

    ants_registration(ref_path, anatomy_file, out_anatomy_path, 'affine')

    out_files = os.listdir(out_anatomy_path)
    out_anatomy_file = [item for item in out_files if 'warped' in item]

    assert len(out_anatomy_file) == 1

    out_anatomy_file = out_anatomy_file[0]
    out_anatomy_file = os.path.join(out_anatomy_path, out_anatomy_file)

    out_mat_file = [item for item in out_files if '.mat' in item][0]
    out_mat_file = os.path.join(out_anatomy_path, out_mat_file)

    # out_deformation_file = [item for item in out_files if 'InverseWarp' in item]
    # if len(out_deformation_file) == 0:
    #     out_deformation_file = None
    # else:
    #     out_deformation_file = out_deformation_file[0]
    #     out_deformation_file = os.path.join(out_anatomy, out_deformation_file)

    for trk in trk_files:
        scil_apply_transform_to_trk(trk, out_anatomy_file, out_mat_file, out_bundles_path) #, out_deformation_file)
        
    return s


def compute_dice_overlap(flip, real, anat):
    
    img = nib.load(anat)
    affine = img.affine
    shape = img.shape


    density_flip = density_map(flip, affine, shape)
    density_real = density_map(real, affine, shape)

    mask_flip = density_flip > 0
    mask_real = density_real > 0

    intersection = np.minimum(density_flip, density_real).sum()
    volume_flip = density_flip.sum()
    volume_real = density_real.sum()

    # intersection = np.logical_and(mask_flip, mask_real).sum()
    # volume_flip = mask_flip.sum()
    # volume_real = mask_real.sum()

  

    if volume_flip + volume_real == 0:
        return 0.0
    
    dice = 2.0 * intersection / (volume_flip + volume_real)
    return dice
        

import numpy as np
from scipy.spatial import KDTree



# def compute_bundle_minimum_distance(streamlines_flipped, streamlines_original):
#     """
#     Calcola la distanza minima assoluta in millimetri tra due bundle e 
#     restituisce le coordinate esatte del punto di massimo contatto.
    
#     Parametri:
#     streamlines_flipped: Lista o ArraySequence di coordinate (es. AF_L flippato)
#     streamlines_original: Lista o ArraySequence di coordinate (es. AF_R originale)
    
#     Ritorna:
#     min_distance (float): La distanza minima in millimetri.
#     point_flipped (np.ndarray): Coordinate [x, y, z] sul bundle flippato.
#     point_original (np.ndarray): Coordinate [x, y, z] sul bundle originale controlaterale.
#     """
#     try:
#         # Appiattiamo le streamline in due enormi array di coordinate N x 3
#         points_flipped = np.vstack(streamlines_flipped)
#         points_original = np.vstack(streamlines_original)
#     except ValueError:
#         print("Errore: uno dei due bundle è vuoto.")
#         return np.inf, None, None

#     # Costruiamo il KDTree sul bundle originale
#     tree_original = KDTree(points_original)

#     # Interroghiamo l'albero: 
#     # - 'distances' conterrà la distanza dal punto più vicino per ogni punto flippato
#     # - 'indices' conterrà l'indice (riga) di quel punto più vicino in points_original
#     distances, indices = tree_original.query(points_flipped, k=1, workers=-1)

#     # 1. Troviamo la posizione (indice) della distanza minima assoluta nell'array
#     min_idx_flipped = np.argmin(distances)
    
#     # 2. Estraiamo il valore della distanza minima
#     min_distance = distances[min_idx_flipped]

#     # 3. Estraiamo le coordinate esatte [x, y, z] per il bundle flippato
#     point_flipped = points_flipped[min_idx_flipped]
    
#     # 4. Usiamo l'array 'indices' per trovare l'indice corrispondente nel bundle originale
#     # ed estrarre le sue coordinate esatte [x, y, z]
#     min_idx_original = indices[min_idx_flipped]
#     point_original = points_original[min_idx_original]

#     return min_distance, point_flipped, point_original



def greedy_assignment(dist_matrix):
    """
    Esegue un'assegnazione greedy 1-a-1 basata sulla distanza minima.
    Restituisce una lista di tuple (indice_bundle_A, indice_bundle_B).
    """
    n_a, n_b = dist_matrix.shape
    assigned_a = set()
    assigned_b = set()
    pairs = []

    # Appiattisce la matrice e ottiene gli indici ordinati dalla distanza minima alla massima
    flat_indices = np.argsort(dist_matrix, axis=None)

    for flat_idx in flat_indices:
        # Riconverte l'indice flat nelle coordinate 2D originali (riga, colonna)
        i, j = np.unravel_index(flat_idx, dist_matrix.shape)

        # Se nessuna delle due streamline è già stata assegnata, crea la coppia
        if i not in assigned_a and j not in assigned_b:
            pairs.append((i, j))
            assigned_a.add(i)
            assigned_b.add(j)

            # Ottimizzazione: fermati quando hai accoppiato tutte le streamline 
            # del bundle più piccolo (i bundle potrebbero avere dimensioni diverse)
            if len(assigned_a) == min(n_a, n_b):
                break

    return pairs


# import numpy as np
# from scipy.spatial import KDTree
# import os
# from dipy.io.stateful_tractogram import StatefulTractogram
# from dipy.io.streamline import save_tractogram


# """
# find_closest_streamline_pair.py
# --------------------------------
# Trova la coppia di streamline più vicine tra due bundle:
# tipicamente un bundle flippato (es. AF_L specchiato) e il suo
# controlaterale originale (es. AF_R).
 
# La ricerca è basata sulla **minima distanza punto-a-punto** tra
# qualsiasi punto del primo bundle e qualsiasi punto del secondo,
# usando un KD-tree per efficienza.
 
# Dipendenze: numpy, scipy, dipy, nibabel
# """
 
# import numpy as np
# from scipy.spatial import cKDTree
# from dipy.io.stateful_tractogram import StatefulTractogram, Space
# from dipy.io.streamline import save_tractogram
 
 

# def _extract_streamlines(bundle):
#     """
#     Estrae le streamline e l'eventuale SFT da un input flessibile.
 
#     Parameters
#     ----------
#     bundle : StatefulTractogram | list | ArraySequence
#         Bundle in ingresso.
 
#     Returns
#     -------
#     streamlines : list of np.ndarray
#     sft : StatefulTractogram | None
#         None se l'input non era un SFT (utile per preservare l'header
#         al momento del salvataggio).
#     """
#     if isinstance(bundle, StatefulTractogram):
#         return list(bundle.streamlines), bundle
#     streamlines = list(bundle)
#     if not streamlines:
#         raise ValueError("Il bundle in ingresso non contiene streamline.")
#     return streamlines, None
 
 
# def _resample(streamlines, n_points):
#     """Ricampiona le streamline a n_points punti (solo per il confronto)."""
#     if n_points is None:
#         return [np.asarray(sl, dtype=np.float64) for sl in streamlines]
#     from dipy.tracking.streamline import set_number_of_points
#     return [np.asarray(set_number_of_points(sl, n_points), dtype=np.float64)
#             for sl in streamlines]
 
 
# def _build_tree(streamlines):
#     """
#     Impila tutti i punti di un bundle in un unico array e costruisce
#     il KD-tree, mantenendo un vettore di appartenenza per streamline.
 
#     Returns
#     -------
#     tree      : cKDTree
#     pts_all   : np.ndarray  (N_tot_punti, 3)
#     sl_ids    : np.ndarray  (N_tot_punti,)  — indice streamline per ogni punto
#     """
#     pts_list, id_list = [], []
#     for j, sl in enumerate(streamlines):
#         arr = np.asarray(sl, dtype=np.float64)
#         pts_list.append(arr)
#         id_list.append(np.full(len(arr), j, dtype=np.int64))
#     pts_all = np.vstack(pts_list)
#     sl_ids  = np.concatenate(id_list)
#     return cKDTree(pts_all), pts_all, sl_ids
 
 
# def _make_sft(streamline, source_sft, reference):
#     """Crea un SFT contenente una singola streamline."""
#     if source_sft is not None:
#         return StatefulTractogram(
#             [streamline],
#             source_sft,
#             space=source_sft.space,
#         )
#     if reference is not None:
#         return StatefulTractogram(
#             [streamline],
#             reference,
#             space=Space.RASMM,
#         )
#     raise ValueError(
#         "Fornire un 'reference' (immagine o path) quando l'input "
#         "non è un StatefulTractogram."
#     )
 
 

 
# def find_closest_streamline_pair(
#     bundle1,
#     bundle2,
#     n_points=20,
#     b1_name=None,
#     b2_name=None,
#     sub_name=None,
#     output_prefix=None,
#     reference=None,
#     verbose=True,
# ):
#     """
#     Trova la coppia di streamline — una da bundle1, una da bundle2 —
#     i cui punti si avvicinano di più nello spazio 3-D.
 
#     Uso tipico: confronto tra un bundle flippato (es. AF_L specchiato
#     in RAS) e il suo controlaterale originale (es. AF_R).
 
#     Parameters
#     ----------
#     bundle1 : StatefulTractogram | list | ArraySequence
#         Primo bundle (es. AF_L flippato).
#     bundle2 : StatefulTractogram | list | ArraySequence
#         Secondo bundle (es. AF_R originale).
#     n_points : int | None
#         Numero di punti a cui ricampionare le streamline prima
#         del confronto (default 20).
#         - Velocizza la ricerca e uniforma la densità dei punti.
#         - Le streamline originali (non ricampionate) vengono comunque
#           restituite e salvate.
#         - Impostare None per usare i punti originali (più lento).
#     output_prefix : str | None
#         Se fornito, salva le due streamline più vicine come:
#           ``<output_prefix>_bundle1.trk``
#           ``<output_prefix>_bundle2.trk``
#     reference : str | Nifti1Image | None
#         Immagine anatomica di riferimento per il salvataggio .trk.
#         Necessaria solo quando l'input non è un SFT.
#     verbose : bool
#         Stampa informazioni sull'avanzamento.
 
#     Returns
#     -------
#     result : dict
#         Chiavi:
#         ``'streamline1'``    — np.ndarray, streamline più vicina da bundle1
#         ``'streamline2'``    — np.ndarray, streamline più vicina da bundle2
#         ``'idx1'``           — int, indice in bundle1
#         ``'idx2'``           — int, indice in bundle2
#         ``'min_distance'``   — float, distanza minima in mm tra la coppia
#         ``'closest_point1'`` — np.ndarray (3,), punto su sl1 più vicino a sl2
#         ``'closest_point2'`` — np.ndarray (3,), punto su sl2 più vicino a sl1
 
#     Notes
#     -----
#     Algoritmo
#     ~~~~~~~~~
#     1. Costruisce un KD-tree con tutti i punti di bundle2.
#     2. Per ogni streamline di bundle1, interroga il KD-tree per trovare
#        il punto più vicino in bundle2 → distanza e streamline di appartenenza.
#     3. Aggiorna il minimo globale.
#     Complessità: O(N1 · P · log(N2 · P)) vs O(N1 · N2 · P²)
#     del doppio ciclo naïve.
#     """
#     # ---- estrazione --------------------------------------------------------
#     streamlines1, sft1 = _extract_streamlines(bundle1)
#     streamlines2, sft2 = _extract_streamlines(bundle2)
 
#     if verbose:
#         print(f"Bundle 1: {len(streamlines1)} streamline")
#         print(f"Bundle 2: {len(streamlines2)} streamline")
 
#     # ---- ricampionamento (solo per il confronto) ---------------------------
#     cmp1 = _resample(streamlines1, n_points)
#     cmp2 = _resample(streamlines2, n_points)
 
#     # ---- KD-tree su bundle2 ------------------------------------------------
#     if verbose:
#         n_pts = sum(len(sl) for sl in cmp2)
#         print(f"Costruzione KD-tree con {n_pts} punti da bundle2 …")
 
#     tree2, pts2_all, sl2_ids = _build_tree(cmp2)
 
#     # ---- ricerca del minimo globale ----------------------------------------
#     global_min         = np.inf
#     best_i = best_j    = 0
#     best_pt1_local_idx = 0          # indice locale in cmp1[best_i]
#     best_pt2_global_idx= 0          # indice globale in pts2_all
 
#     for i, sl1 in enumerate(cmp1):
#         # Nearest-neighbour per ogni punto di sl1 nell'intero bundle2
#         dists, nn_idx = tree2.query(sl1)          # shape (P,)
 
#         local_argmin  = int(np.argmin(dists))
#         local_min_d   = float(dists[local_argmin])
 
#         if local_min_d < global_min:
#             global_min          = local_min_d
#             best_i              = i
#             best_pt1_local_idx  = local_argmin
#             best_pt2_global_idx = int(nn_idx[local_argmin])
#             best_j              = int(sl2_ids[best_pt2_global_idx])
 
#     if verbose:
#         print(f"\nCoppia trovata → bundle1[{best_i}]  ↔  bundle2[{best_j}]")
#         print(f"Distanza minima: {global_min:.4f} mm")
 
#     # ---- streamline originali e punti più vicini ----------------------------
#     sl_best1 = np.asarray(streamlines1[best_i], dtype=np.float64)
#     sl_best2 = np.asarray(streamlines2[best_j], dtype=np.float64)
 
#     closest_pt1 = cmp1[best_i][best_pt1_local_idx]
#     closest_pt2 = pts2_all[best_pt2_global_idx]

#     np.save(f"/home/alessia.ianes/Desktop/data/TractoInferno_rearranged/correspondences/{sub_name}_{b1_name}_{b2_name}_closest_points.npy", np.array([closest_pt1, closest_pt2]))
#     np.save(f"/home/alessia.ianes/Desktop/data/TractoInferno_rearranged/correspondences/{sub_name}_{b1_name}_{b2_name}_min_distance.npy", global_min)

 
#     # ---- salvataggio --------------------------------------------------------
#     if output_prefix is not None:
#         _save_pair(
#             sl_best1, sl_best2,
#             sft1, sft2, b1_name, b2_name, sub_name,
#             reference,
#             output_prefix,
#             verbose,
#         )
 
#     return {
#         "streamline1":    sl_best1,
#         "streamline2":    sl_best2,
#         "idx1":           best_i,
#         "idx2":           best_j,
#         "min_distance":   global_min,
#         "closest_point1": closest_pt1,
#         "closest_point2": closest_pt2,
#     }
 
 
# # ---------------------------------------------------------------------------
# # Salvataggio
# # ---------------------------------------------------------------------------
 
# def _save_pair(sl1, sl2, sft1, sft2, b1_name, b2_name, sub_name, reference, prefix, verbose=True):
#     """
#     Salva le due streamline come file .trk separati.
 
#     Riutilizza l'header dell'SFT originale quando disponibile;
#     altrimenti usa il 'reference' fornito con spazio RASMM.
#     """
#     pairs = [
#         ('flip', sl1, sft1),
#         ('original', sl2, sft2),
#     ]
#     for label, sl, sft in pairs:
#         sft_out = _make_sft(sl, sft, reference)
#         path = f"/home/alessia.ianes/Desktop/data/TractoInferno_rearranged/correspondences_trk/{sub_name}_{prefix}_{b1_name}_{b2_name}_{label}_bundle.trk"
#         save_tractogram(sft_out, path, bbox_valid_check=False)
#         if verbose:
#             print(f"Salvato → {path}")

    
# def find_correspondence(root_path, flipped_tractogram, original_bundle, bundles, opp_name, sub):

#     # print(matched_pairs)
#     corr = os.path.join(root_path, 'correspondences')
#     os.makedirs(corr, exist_ok=True)
#     corr_trk = os.path.join(root_path, 'correspondences_trk')
#     os.makedirs(corr_trk, exist_ok=True)

#     # --- ricerca e salvataggio ----------------------------------------------
#     result = find_closest_streamline_pair(
#         bundle1       = flipped_tractogram,   # o una lista di streamline
#         bundle2       = original_bundle,            # o una lista di streamline
#         n_points      = 32,                  # punti per il confronto
#         b1_name       = bundles,                 # solo per il salvataggio
#         b2_name       = opp_name,                 # solo per il salvataggio
#         sub_name     = sub,                       # solo per il salvataggio
#         output_prefix = "closest_pair",      # → closest_pair_bundle1/2.trk
#         # reference   = "T1.nii.gz"         # solo se i bundle NON sono SFT
#     )

#     print("\n--- Risultati ---")
#     print(f"Indice in bundle1 : {result['idx1']}")
#     print(f"Indice in bundle2 : {result['idx2']}")
#     print(f"Distanza minima   : {result['min_distance']:.4f} mm")
#     print(f"Punto su sl1      : {result['closest_point1']}")
#     print(f"Punto su sl2      : {result['closest_point2']}")

#     # Accesso diretto alle streamline
#     # sl1 = result["streamline1"]   # np.ndarray (N_pts, 3)
#     # sl2 = result["streamline2"]   # np.ndarray (M_pts, 3)



    

from dipy.tracking.distances import bundles_distances_mdf

def flip_bundle(bundle_path, root_path):

    for bundles in sorted(os.listdir(bundle_path)):
      
        for set_f in sorted(os.listdir(os.path.join(bundle_path, bundles))):
            for sub in sorted(os.listdir(os.path.join(bundle_path, bundles, set_f))):
                for file in os.listdir(os.path.join(bundle_path, bundles, set_f, sub)):

                    if os.path.exists(os.path.join(root_path, 'flip_bundles', bundles, set_f, sub, file.replace('_32_points.trk', '_flipped.trk'))):
                        continue

                    file_path = os.path.join(bundle_path, bundles, set_f, sub, file)




                    if '_L' in bundles:
                        opp_name = bundles.replace('_L', '_R')
                        opp_path = os.path.join(bundle_path, bundles.replace('_L', '_R'), set_f, sub, file.replace('_L', '_R'))
                    elif '_R' in bundles:
                        opp_name = bundles.replace('_R', '_L')

                        opp_path = os.path.join(bundle_path, bundles.replace('_R', '_L'), set_f, sub, file.replace('_R', '_L'))
                    else:
                        continue

                    ref_anat = [os.path.join(root_path, 'anat', set_f, sub, anat) for anat in os.listdir(os.path.join(root_path, 'anat', set_f, sub))][0]

                    trk = load_tractogram(file_path, ref_anat, bbox_valid_check=False)
                    streamlines = trk.streamlines

                    flipped_streamlines = []
                    
                    print("Executing flipping on x axis ...")
                    for sl in streamlines:
                        sl_flipped = sl.copy()
                        sl_flipped[:, 0] = -sl_flipped[:, 0]

                        flipped_streamlines.append(sl_flipped)
                    
                    flipped_tractogram = StatefulTractogram.from_sft(flipped_streamlines, trk)#, ref_anat, Space.RASMM)

                    print(f"Saving bundle {bundles} for sub {sub}... ")

                    original_bundle = load_tractogram(opp_path, ref_anat, bbox_valid_check=False)
                    
                    saving_path = os.path.join(root_path, 'flip_bundles', bundles, set_f, sub)
                    os.makedirs(saving_path, exist_ok=True)
                    save_tractogram(flipped_tractogram, f"{saving_path}/{file.replace('_32_points.trk', '_flipped.trk')}", bbox_valid_check=False)
                    print(f"Saving bundle {bundles} for sub {sub} completed")
                    
                    # print(f"Creating correspondence between flipped {bundles} and {opp_name} for sub {sub} in {set_f}")

                    # dist_matrix = bundles_distances_mdf(flipped_tractogram.streamlines, original_bundle.streamlines)
                    # matched_pairs = greedy_assignment(dist_matrix)

                    # save_corr = os.path.join(root_path, 'correspondences')
                    # os.makedirs(save_corr, exist_ok=True)
                    # np.save(f"{save_corr}/{file.replace('_32_points.trk', f'_to_{opp_name}_matched_pairs.npy')}", matched_pairs)

                    # find_correspondence(root_path, flipped_tractogram, original_bundle, bundles, opp_name, sub)

                    


    
def check_affine(path):
    sft = load_tractogram(path, 'same', bbox_valid_check=False)  
    print(sft.affine)


def assess_correspondence(initial_path, target_path, ref_anat, csv_file):
    # check = [os.path.join(out_path, 'bundles', folder_set, sub) for sub in initial_path]
    target_paths = [os.path.join(target_path, file) for file in sorted(os.listdir(target_path))]
    target_names = [os.path.basename(target).split('.')[0].upper() for target in target_paths]

    print(target_names)
    
    
    with open(csv_file, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['Subject', 'TractoInferno Bundle', 'Atlas Bundle', 'Dice Score'])    

        for sub in initial_path:
            check = os.path.join(out_path, 'bundles', folder_set, sub)
            # print(sub)
            for bundle in sorted(os.listdir(check)):
                
                

                # print(bundle)
                bundle_name = bundle.split('__')[-1].split('_aligned')[0].upper()
                print(bundle_name)


                if bundle_name not in target_names:
                    continue

                file_path = os.path.join(check, bundle)
                target_name_out = [target for target in target_names if bundle_name in target.upper()][0]
                print(target_name_out)
                
                target_bundle_path = [target for target in target_paths if bundle_name in target.upper()][0]
                
              

                print(file_path)
                print(target_bundle_path)

                trk_check = load_tractogram(file_path, ref_anat, bbox_valid_check=False)
                trk_target = load_tractogram(target_bundle_path, ref_anat, bbox_valid_check=False)

                dice_score = compute_dice_overlap(trk_check.streamlines, trk_target.streamlines, ref_anat)

                writer.writerow([sub, bundle_name, target_name_out, f'{dice_score:.4f}'])




from sklearn.metrics import pairwise_distances, pairwise_distances_argmin_min
def dataset_reduction(path, anat, save_folder, threshold):
    # bundles_embs --> AF_L --> testset --> sub ....
    
    for bundle in sorted(os.listdir(path)):
        for set_f in sorted(os.listdir(os.path.join(path, bundle))):
           
            print(set_f)

            for sub in sorted(os.listdir(os.path.join(path, bundle, set_f))):
                print(sub)
                start = perf_counter()

                for file in sorted(os.listdir(os.path.join(path, bundle, set_f, sub))):
                    print(file)
                    file_path = os.path.join(path, bundle, set_f, sub, file)
                    print(os.path.join(save_folder, bundle, set_f, sub, file.replace('.trx', f'_{bundle}_REDUCED.trk')))
                    if os.path.exists(os.path.join(save_folder, bundle, set_f, sub, file.replace('.trx', f'_{bundle}_REDUCED.trk'))):
                        print('EXISTS')
                        continue

                    bundle_name = bundle.upper()
                    if '_L' in bundle_name:
                        opposite_bundle_path = os.path.join(path, bundle.replace('_L', '_R'), set_f, sub, file)
                        opposite_bundle_name = bundle.replace('_L', '_R').upper()
                    elif '_R' in bundle_name:
                        opposite_bundle_path = os.path.join(path, bundle.replace('_R', '_L'), set_f, sub, file)
                        opposite_bundle_name = bundle.replace('_R', '_L').upper()

                    print(opposite_bundle_path)


                    file_trx = load_tractogram(file_path, anat, bbox_valid_check=False)
                    emb_file = np.array(file_trx.data_per_streamline['embeddings']).astype(np.float32)
                


                    opposite_trx = load_tractogram(opposite_bundle_path, anat, bbox_valid_check=False)
                    emb_opposite = np.array(opposite_trx.data_per_streamline['embeddings']).astype(np.float32)   


                    

                    bundle_sl = np.asarray(file_trx.streamlines, dtype=object)
                    opposite_sl = np.asarray(opposite_trx.streamlines, dtype=object)

                
                    n_prototypes = min(len(bundle_sl), len(opposite_sl), threshold)
                    print(n_prototypes)


                    # WITH AGGLOMERATIVE CLUSTERING
                    # agg_clustering = AgglomerativeClustering(
                    #     n_clusters=n_prototypes

                    # )

                    


                    # 2. Initialize agglomerative clustering given the matrix already computed
                    agg_clustering = AgglomerativeClustering(
                        n_clusters=n_prototypes,
                        metric='precomputed',
                        linkage='average' 
                    )

                    dist_matrix = pairwise_distances(emb_file, metric='euclidean', n_jobs=100)
                    dist_matrix_opposite = pairwise_distances(emb_opposite, metric='euclidean', n_jobs=100)

                    print("Matrices computed! Starting clustering...")

                    reduced_file = agg_clustering.fit_predict(dist_matrix)
                    reduced_opposite = agg_clustering.fit_predict(dist_matrix_opposite)


                    sl_b = []
                    sl_opp = []

                    for i in range(n_prototypes):
                        # 1. Trova tutte le posizioni (indici) delle streamline finite nel cluster 'i'
                        idx = np.where(reduced_file == i)[0]
                        idx_opp = np.where(reduced_opposite == i)[0]
                        
                        # 2. Isola i dati (le coordinate/features) solo di queste streamline
                        X_cluster = emb_file[idx]
                        X_opposite = emb_opposite[idx_opp]

                        
                        # 3. Calcola il "centro" matematico di questo gruppo facendone la media
                        centroid = np.mean(X_cluster, axis=0).reshape(1, -1)
                        centroid_opp = np.mean(X_opposite, axis=0).reshape(1, -1)

                        
                        # 4. Trova quale streamline REALE di questo gruppo è più vicina al centroide
                        # pairwise_distances_argmin_min restituisce l'indice dell'elemento più vicino
                        closer, _ = pairwise_distances_argmin_min(centroid, X_cluster)
                        closer_opp, _ = pairwise_distances_argmin_min(centroid_opp, X_opposite)

                        
                        # 5. Risali all'indice originale e salvalo nella lista finale
                        original_idx = idx[closer[0]]
                        original_idx_opp = idx_opp[closer_opp[0]]

                        sl_b.append(original_idx)
                        sl_opp.append(original_idx_opp)


                    print(f"Number of extracted indexes: {len(sl_b)}")

                    # sl_pred = [file_trx.streamlines[label] for label in np.unique(reduced_file)]
                    sl_pred = [file_trx.streamlines[label] for label in sl_b]

                    new_tractogram = StatefulTractogram.from_sft(sl_pred, file_trx)

                    # sl_pred = [opposite_trx.streamlines[label] for label in np.unique(reduced_opposite)]
                    sl_pred = [opposite_trx.streamlines[label] for label in sl_opp]

                    new_tractogram_opposite = StatefulTractogram.from_sft(sl_pred, opposite_trx)
                        






                    # WITH FFT
                    # print(f"Computing prototypes for {bundle_name}")
                    # prototypes_bundle = compute_dissimilarity(emb_file, euclidean_distance, "fft", n_prototypes)
                    # new_streamlines = file_trx.streamlines[prototypes_bundle.tolist()]
                    # new_tractogram = StatefulTractogram.from_sft(new_streamlines, file_trx)


                    # print(f"Computing prototypes for {opposite_bundle_name}")
                    # prototypes_opposite = compute_dissimilarity(emb_opposite, euclidean_distance, "fft", n_prototypes)
                    # new_streamlines_opposite = opposite_trx.streamlines[prototypes_opposite.tolist()]
                    # new_tractogram_opposite = StatefulTractogram.from_sft(new_streamlines_opposite, opposite_trx)



                    # # RANDOM
                    # # prototypes_bundle = np.random.choice(bundle_sl, n_prototypes)
                    # new_tractogram = StatefulTractogram.from_sft(prototypes_bundle, file_trx)

                    # # prototypes_opposite = np.random.choice(opposite_sl, n_prototypes)
                    # new_tractogram_opposite = StatefulTractogram.from_sft(prototypes_opposite, opposite_trx)


                    save_path = os.path.join(save_folder, bundle, set_f, sub)
                    os.makedirs(save_path, exist_ok=True)
                
                    save_tractogram(new_tractogram, os.path.join(save_path, file.replace('.trx', f'_{bundle_name}_REDUCED.trk')), bbox_valid_check=False)

                    save_path_opposite = os.path.join(save_folder, opposite_bundle_name, set_f, sub)
                    os.makedirs(save_path_opposite, exist_ok=True)

                    save_tractogram(new_tractogram_opposite, os.path.join(save_path_opposite, os.path.basename(opposite_bundle_path).replace('.trx', f'_{opposite_bundle_name}_REDUCED.trk')), bbox_valid_check=False)
                    end = perf_counter()

                    txt_time = os.path.join(save_folder.replace('reduced_bundles', 'time_reduction'), sub)
                    os.makedirs(txt_time, exist_ok=True)
                    print(f"Time for reducing bundle {bundle_name}, {opposite_bundle_name} for sub {sub}: {end-start}")
                    with open(f'{txt_time}/{bundle_name}_{opposite_bundle_name}_time_reduction.txt', 'w') as f:
                        f.write(f'Time for generating prototypes {end-start}')

            
            







    # start_sub = perf_counter()

    # for sub in tqdm(sorted(os.listdir(set_path)), total=len(os.listdir(set_path))):
    #     sub_folder = os.path.join(set_path, sub)
    #     print(f"{sub}")

    #     for file in sorted(os.listdir(sub_folder)):

    #         if os.path.exists(os.path.join(save_folder, sub, file.replace('.trk', '_REDUCED.trk'))):
    #             continue

    #         file_path = os.path.join(sub_folder, file)
    #         bundle_name = file.split('__')[-1].split('_aligned')[0].upper()
    #         if '_L' in bundle_name:
    #             opposite_bundle_path = os.path.join(sub_folder, file.replace('_L', '_R'))
    #         elif '_R' in bundle_name:
    #             opposite_bundle_path = os.path.join(sub_folder, file.replace('_R', '_L'))

    #         file_trx = load_tractogram(file_path, anat, bbox_valid_check=False)
    #         opposite_trx = load_tractogram(opposite_bundle_path, anat, bbox_valid_check=False)

    #         bundle_sl = np.asarray(file_trx.streamlines, dtype=object)
    #         opposite_sl = np.asarray(opposite_trx.streamlines, dtype=object)

           
    #         n_prototypes = min(len(bundle_sl), len(opposite_sl), threshold)


    #         # print(bundle_name, os.path.basename(opposite_bundle_path).split('__')[-1].split('_aligned')[0].upper(), sub)
    #         # print(bundle_name, len(bundle_sl))
    #         # print(os.path.basename(opposite_bundle_path).split('__')[-1].split('_aligned')[0].upper(), len(opposite_sl))
    #         # print(n_prototypes)


    #         # start = perf_counter()


    #         # WITH FFT
    #         print(f"Computing prototypes for {bundle_name}")
    #         prototypes_bundle = compute_dissimilarity(bundle_sl, 'euclidean', "fft", n_prototypes)

    #         print(f"Computing prototypes for {os.path.basename(opposite_bundle_path).split('__')[-1].split('_aligned')[0].upper()}")
    #         prototypes_opposite = compute_dissimilarity(opposite_sl, 'euclidean', "fft", n_prototypes)

    #         end = perf_counter()

    #         # RANDOM
    #         # prototypes_bundle = np.random.choice(bundle_sl, n_prototypes)
    #         new_tractogram = StatefulTractogram.from_sft(prototypes_bundle, file_trk)

    #         # prototypes_opposite = np.random.choice(opposite_sl, n_prototypes)
    #         new_tractogram_opposite = StatefulTractogram.from_sft(prototypes_opposite, opposite_trk)

            


    #         save_path = os.path.join(save_folder, sub)
    #         os.makedirs(save_path, exist_ok=True)
        
    #         save_tractogram(new_tractogram, os.path.join(save_path, file.replace('.trk', '_REDUCED.trk')), bbox_valid_check=False)

    #         save_path_opposite = os.path.join(save_folder, sub)
    #         os.makedirs(save_path_opposite, exist_ok=True)

    #         save_tractogram(new_tractogram_opposite, os.path.join(save_path_opposite, os.path.basename(opposite_bundle_path).replace('.trk', '_REDUCED.trk')), bbox_valid_check=False)
            
    #         # # RESAMPLE
    #         # start = perf_counter()

    #         # cmd_trk = ['scil_tractogram_resample', '--never_upsample', '-f', file_path, str(n_prototypes), os.path.join(save_path, file.replace('.trk', '_REDUCED.trk'))]
    #         # subprocess.check_call(cmd_trk, stdout=subprocess.DEVNULL)

    #         # cmd_opposite = ['scil_tractogram_resample', '--never_upsample', '-f', opposite_bundle_path, str(n_prototypes), os.path.join(save_path_opposite, os.path.basename(opposite_bundle_path).replace('.trk', '_REDUCED.trk'))]
    #         # subprocess.check_call(cmd_opposite, stdout=subprocess.DEVNULL)
            
    #         # end = perf_counter()
            
    #         # print(f"It took {end-start} s to compute and save {bundle_name} and {os.path.basename(opposite_bundle_path).split('__')[-1].split('_aligned')[0].upper()} prototypes")
    #         # with open(os.path.join(save_folder, sub, os.path.basename(opposite_bundle_path).replace('.trk', '_REDUCED.trk')), 'w') as f:
    #         #     f.write(f'Time for generating prototypes {end-start}')
        #     print(f"Time for reducing bundle {bundle_name}, {os.path.basename(opposite_bundle_path).split('__')[-1].split('_aligned')[0]}: {end-start}")

        # end_sub = perf_counter()
        # print(f"Time for {sub}: {end_sub - start_sub}")
        


           
def smooth_bundles(in_path, out_path, sigma):

    for bundle in sorted(os.listdir(in_path)):
        for set_f in sorted(os.listdir(os.path.join(in_path, bundle))):
            for sub in sorted(os.listdir(os.path.join(in_path, bundle, set_f))):
                out_folder = os.path.join(out_path, bundle, set_f, sub)
                os.makedirs(out_folder, exist_ok=True)
                for file in os.listdir(os.path.join(in_path, bundle, set_f, sub)):
                    file_path = os.path.join(in_path, bundle, set_f, sub, file)

                    file_out_path = os.path.join(out_folder, file.replace('_32_points.trk', '_smooth.trk'))

                    cmd = ['scil_tractogram_smooth', '--gaussian', str(sigma), '-f', file_path, file_out_path]

                    subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
                    


        


def nb_points_reduction(path, save_path, points):
    for bundle in sorted(os.listdir(path)):
        for set_f in sorted(os.listdir(os.path.join(path, bundle))):
            for sub in sorted(os.listdir(os.path.join(path, bundle, set_f))):
                for file in os.listdir(os.path.join(path, bundle, set_f, sub)):
                    file_path = os.path.join(path, bundle, set_f, sub, file)

                    saving_path = os.path.join(save_path, bundle, set_f, sub)
                    os.makedirs(saving_path, exist_ok=True)

                    print(f"Reducing {bundle} for sub {sub} of set {set_f} to {points} points...\n")
                    cmd = ['scil_tractogram_resample_nb_points', '--nb_pts_per_streamline', str(points), '-f', file_path, f"{saving_path}/{file.replace('_reduced.trk', '_32_points.trk')}"]
                    subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
                    print("Resampling done!\n")

   
def merging(path, user):
    # for bundle in os.listdir(path):
    #     bundle_path = os.path.join(path, bundle)
    for set_f in os.listdir(path):
        set_path = os.path.join(path, set_f)
        for sub in os.listdir(set_path):
            sub_path = os.path.join(set_path, sub)
            saving_folder = f'/home/{user}/Desktop/data/TractoInferno_subjects/{set_f}/{sub}'
            os.makedirs(saving_folder, exist_ok=True)
            merge_bundles(sub_path, sub, saving_folder)
                # for file in os.listdir(sub_path):
                #     if not file.endswith('.trx'):
                #         continue
                #     file_path = os.path.join(sub_path, file)

                #     new_folder = '/home/alessia/Desktop/data/Tractoinferno_subjects'
                #     os.makedirs(new_folder, exist_ok=True)

                #     dst = os.path.join(new_folder, set_f, sub)
                #     os.makedirs(dst, exist_ok=True)       


from reduce_bundle import reduce


if __name__ == '__main__':


    user = 'alessia.ianes'

    # emb_path = f'/home/{user}/Desktop/data/TractoInferno_rearranged/bundles_embs'
    # aligned_anat = f'/home/{user}/Desktop/data/TractoInferno_aligned/anat/testset/sub-1006/sub-1006__T1w_affine_warped.nii.gz'
    


    # ====== 1. Reduce number of streamlines
    prototype_threshold = 5000
    path_bundles = f'/home/{user}/Desktop/data/TractoInferno_rearranged/bundles'
    reduced_bundles = f'/home/{user}/Desktop/data/TractoInferno_rearranged/reduced_bundles_dynamic'
    os.makedirs(reduced_bundles, exist_ok=True)

    # reduce(user, path_bundles, reduced_bundles)

    # ====== 2. Resampling number of points per streamline

    reduced_streamlines = f'/home/{user}/Desktop/data/TractoInferno_rearranged/reduced_streamlines'
    os.makedirs(reduced_streamlines, exist_ok=True)
    points = 32
    
    # nb_points_reduction(reduced_bundles, reduced_streamlines, points)


    # ====== 3. Flip bundles to prepare data for correspondence check
    # flip_bundle(reduced_streamlines, f'/home/{user}/Desktop/data/TractoInferno_rearranged')



    # ====== 4. Check correspondence
    # ADD FUNCTION TO CONNECT TO MATCHING.PY


    # ====== 5. Merging trx per subject
    merging(f'/home/{user}/Desktop/data/TractoInferno_rearranged/match_mdf_trx', user)


    # ====== 6. Smoothin bundles
   
    # smooth_bundles_folder = f'/home/{user}/Desktop/data/TractoInferno_rearranged/smooth_bundles'
    # os.makedirs(smooth_bundles_folder, exist_ok=True)

    # sigma = 3

    # smooth_bundles(reduced_bundles, smooth_bundles_folder, sigma)
    
    


   








    # ========================= SLF SCIL UNION =========================
    # slf_i_L = load_tractogram(f'/home/{user}/Desktop/data/atlas_scil/tractograms/SLFI_L.trk', 'same')
    # slf_i_R = load_tractogram(f'/home/{user}/Desktop/data/atlas_scil/tractograms/SLFI_R.trk', 'same')
    # slf_ii_L = load_tractogram(f'/home/{user}/Desktop/data/atlas_scil/tractograms/SLFII_L.trk', 'same')
    # slf_ii_R = load_tractogram(f'/home/{user}/Desktop/data/atlas_scil/tractograms/SLFII_R.trk', 'same')
    # slf_iii_L = load_tractogram(f'/home/{user}/Desktop/data/atlas_scil/tractograms/SLFIII_L.trk', 'same')
    # slf_iii_R = load_tractogram(f'/home/{user}/Desktop/data/atlas_scil/tractograms/SLFIII_R.trk', 'same')

    # slf_L = list(slf_i_L.streamlines) + list(slf_ii_L.streamlines) + list(slf_iii_L.streamlines)
    # slf_R = list(slf_i_R.streamlines) + list(slf_ii_R.streamlines) + list(slf_iii_R.streamlines)

    # slf_L_trk = StatefulTractogram(slf_L, slf_i_L, slf_i_L.space)
    # save_tractogram(slf_L_trk, f'/home/{user}/Desktop/data/atlas_scil/bundles_of_interest/SLF_L.trk')

    # slf_R_trk = StatefulTractogram(slf_R, slf_i_R, slf_i_R.space)
    # save_tractogram(slf_R_trk, f'/home/{user}/Desktop/data/atlas_scil/bundles_of_interest/SLF_R.trk')




    




    # # ===== SETUP FOR BUNDLE SUBDIVISION =====
    # bundle_path = f'/home/{user}/Desktop/data/atlas_scil/t1_and_labels/scil_correspondences.json'
    # streamline_path = f'/home/{user}/Desktop/data/atlas_scil/t1_and_labels/scil_labels.npy'


    # labels_name, id_to_name, streamline_labels = initialization(bundle_path, streamline_path)

    # sft = load_tractogram(f'/home/{user}/Desktop/data/atlas_scil/trk_files/scil_merged_atlas_MNI_152_1mm/scil_merged_atlas_MNI_152_1mm.trk', 'same')
    # divide_in_bundles(streamline_labels, sft, f'/home/{user}/Desktop/data/atlas_scil/tractograms', labels_name)
    
    
    
    
    
    # assess_correspondence(data_path, target_path, aligned_anat, csv_file)
    # flip_bundle(out_path, folder_set)
   


    # ===== REGISTRATION SETUP =====
    # num_workers = 4
    # subs = sorted(os.listdir(os.path.join(dataset_path, 'bundles', folder_set)))
    # print(subs)
    

    # with ProcessPoolExecutor(max_workers=num_workers) as executor:
    #     futures = [executor.submit(process_sub, dataset_path, out_path, ref_path, folder_set, s) for s in subs]

    #     for f in tqdm(as_completed(futures), desc='Alignment progress', total=len(futures)):
    #         s = f.result()
