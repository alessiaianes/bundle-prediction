import os
import subprocess

from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.distance import euclidean

from tqdm import tqdm
import numpy as np
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.tracking.utils import density_map
from sklearn.cluster import AgglomerativeClustering
import csv
import nibabel as nib
from functions import *
from time import perf_counter

def ants_registration(fixed, moving, out_path, transform): # FOR REGISTRATION

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


def process_sub(data_path, out_path, ref_path, set_path, s): # FOR REGISTRATION

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


def compute_dice_overlap(flip, real, anat):# NO MORE USED
    
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



def greedy_assignment(dist_matrix):# NO MORE USED
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


    

from dipy.tracking.distances import bundles_distances_mdf

def flip_bundle(bundle_path, save_path):

    for bundles in sorted(os.listdir(bundle_path)):
      
        for set_f in sorted(os.listdir(os.path.join(bundle_path, bundles))):
            for sub in sorted(os.listdir(os.path.join(bundle_path, bundles, set_f))):
                for file in os.listdir(os.path.join(bundle_path, bundles, set_f, sub)):

                    if os.path.exists(os.path.join(save_path, bundles, set_f, sub, file.replace('_32_points.trk', '_flipped.trk'))):
                        continue

                    file_path = os.path.join(bundle_path, bundles, set_f, sub, file)


                    trk = load_tractogram(file_path, 'same', bbox_valid_check=False)
                    streamlines = trk.streamlines

                    flipped_streamlines = []
                    
                    print("Executing flipping on x axis ...")
                    for sl in streamlines:
                        sl_flipped = sl.copy()
                        sl_flipped[:, 0] = -sl_flipped[:, 0]

                        flipped_streamlines.append(sl_flipped)
                    
                    flipped_tractogram = StatefulTractogram.from_sft(flipped_streamlines, trk)#, ref_anat, Space.RASMM)

                    print(f"Saving bundle {bundles} for sub {sub}... ")

                    
                    saving_path = os.path.join(save_path, bundles, set_f, sub)
                    os.makedirs(saving_path, exist_ok=True)
                    save_tractogram(flipped_tractogram, f"{saving_path}/{file.replace('_32_points.trk', '_flipped.trk')}", bbox_valid_check=False)
                    print(f"Saving bundle {bundles} flipped for sub {sub} completed")
                  
                    


    
def check_affine(path): #NO MORE USED
    sft = load_tractogram(path, 'same', bbox_valid_check=False)  
    print(sft.affine)


def assess_correspondence(initial_path, target_path, ref_anat, csv_file): # NO MORE USED
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
def dataset_reduction(path, anat, save_folder, threshold): # NO MORE USED
    
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
    for set_f in os.listdir(path):
        set_path = os.path.join(path, set_f)
        for sub in os.listdir(set_path):
            sub_path = os.path.join(set_path, sub)
            saving_folder = f'/home/{user}/Desktop/data/TractoInferno_subjects/{set_f}/{sub}'
            os.makedirs(saving_folder, exist_ok=True)
            merge_bundles(sub_path, sub, saving_folder)
                  


from reduce_bundle import reduce
from matching import match_streamlines


if __name__ == '__main__':


    user = 'alessia.ianes'
    


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
    # flip_folder = f'/home/{user}/Desktop/data7TractoInferno_rearranged/flip_bundles'
    # os.makedirs(flip_folder, exist_ok=True)
    # flip_bundle(reduced_streamlines, flip_folder)



    # ====== 4. Check correspondence
    # match_streamlines(user, reduced_streamlines, flip_folder)



    # ====== 5. Merging trx per subject
    merging(f'/home/{user}/Desktop/data/TractoInferno_rearranged/match_mdf_trx', user)


    # ====== 6. Smoothing bundles
   
    # smooth_bundles_folder = f'/home/{user}/Desktop/data/TractoInferno_rearranged/smooth_bundles'
    # os.makedirs(smooth_bundles_folder, exist_ok=True)

    # sigma = 3

    # smooth_bundles(reduced_bundles, smooth_bundles_folder, sigma)
    
    


   
# ==============================================================================================================================================================






    # ===== REGISTRATION SETUP =====
    # num_workers = 4
    # subs = sorted(os.listdir(os.path.join(dataset_path, 'bundles', folder_set)))
    # print(subs)
    

    # with ProcessPoolExecutor(max_workers=num_workers) as executor:
    #     futures = [executor.submit(process_sub, dataset_path, out_path, ref_path, folder_set, s) for s in subs]

    #     for f in tqdm(as_completed(futures), desc='Alignment progress', total=len(futures)):
    #         s = f.result()
