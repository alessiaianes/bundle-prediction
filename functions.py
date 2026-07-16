import os
import json
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nib

from dipy.tracking.utils import density_map
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram



def compute_dice_overlap(flip, real, anat):# UESD ONLY FOR AN INITIAL CHECK
    """
    Compute dice coefficient between 2 bundles.
    """
    
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




def flip_bundle(bundle_path, save_path):
    """
    Flip all bundles in the given path and save them.
    """

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





def pairwise_mdf(
        streamlines: np.ndarray,
        prototypes: np.ndarray,
) -> np.ndarray:
    """
    Computes the pairwise Minimum Average Direct-Flip (MDF) distance.

    Args:
        streamlines: (B, S, N, 3)
        prototypes:  (B, P, N, 3)

    Returns:
        mdf: (B, S, P)
    """
    print(streamlines.shape)
    # Expand dimensions for broadcasting
    # (B, S, 1, N, 3)
    streamlines = streamlines[:, np.newaxis, :, :]

    # (B, 1, P, N, 3)
    prototypes =  prototypes[np.newaxis, :, :, :]


    # ---------- Forward distance ----------
    forward = np.linalg.norm(
        streamlines - prototypes,
        axis=-1
    ).mean(axis=-1)  # (B, S, P)

    # ---------- Flipped distance ----------
    prototypes_flip = np.flip(prototypes, axis=2)

    backward = np.linalg.norm(
        streamlines - prototypes_flip,
        axis=-1
    ).mean(axis=-1)  # (B, S, P)



    # MDF
    return np.minimum(forward, backward)




def merge_bundles(path, sub, save_path):
    """
    Create a single trx from all bundles of a subject.
    """
	
    files = sorted(os.listdir(path))
    files = [item for item in files if 'trx' in item]
	
    correspondeces = dict()
    sc_correspondeces = dict()
	


    bundles = [bundle.split('_matched')[0] for bundle in files]
	
    all_streamlines = list()
    all_labels = list()
	
    total_bundles = len(bundles)
    offset = 0
	
    for idx, b in enumerate(bundles):
        print(b)
        offset += 1
        correspondeces[idx] = b
        sc_correspondeces[idx] = b
        trx_file = [item for item in files][idx]

        print(trx_file)
        sft = load_tractogram(os.path.join(path, trx_file), reference='same')
        all_streamlines.extend(list(sft.streamlines))
        labels = [idx] * len(sft.streamlines)
        all_labels.extend(labels)
		
        if idx == total_bundles - 1:
            merged_atlas = StatefulTractogram.from_sft(all_streamlines, sft)
          
            save_tractogram(merged_atlas, f'{save_path}/{sub}_merged.trx')
            np.save(f'{save_path}/{sub}_labels.npy', np.array(all_labels))
            with open(f'{save_path}/{sub}_correspondences.json', 'w') as target:
                json.dump(correspondeces, target)


def merging(path, user):
    """
    Calls "merge_bundles" to create a single trx per subject.
    """
    for set_f in os.listdir(path):
        set_path = os.path.join(path, set_f)
        for sub in os.listdir(set_path):
            sub_path = os.path.join(set_path, sub)
            saving_folder = f'/home/{user}/Desktop/data/TractoInferno_subjects/{set_f}/{sub}'
            os.makedirs(saving_folder, exist_ok=True)
            print(f"Merging new trx for sub {sub}...")
            merge_bundles(sub_path, sub, saving_folder)



def rearrange_dataset(origin, dest):
    """
    Rearrange the dataset.
    Origin structure: root -> set -> sub -> bundles trk
    Dest structure: root -> bundle -> set -> sub -> single bundle trk
    """ 
    for set_f in os.listdir(origin):
        for sub in tqdm(sorted(os.listdir(os.path.join(origin, set_f))), total=len(os.listdir(os.path.join(origin, set_f)))):
            for file in os.listdir(os.path.join(origin, set_f, sub)):
                dst = os.path.join(dest, file.split('__')[-1].split('_aligned')[0], set_f, sub)
                os.makedirs(dst, exist_ok=True)

    for sub in sorted(os.listdir(dest)):
        src = os.path.join(origin, sub)
        for file in os.listdir(src):
            shutil.copy(os.path.join(src, file), os.path.join(dest, sub))


# ========== FIRST STEPS ON TRACTOINFERNO DATASET ==========


def check_for_anat(path_anat, ok_anat, miss_anat, copy_dst):
    """
    Check if anat file is there for a particular path.
    """
    init_anat = ok_anat
    for f in os.listdir(path_anat):
        if f.endswith('.nii') or f.endswith('.nii.gz'):
            # shutil.copy(os.path.join(path_anat, f), copy_dst)
            ok_anat += 1
            break
        
    if ok_anat == init_anat:
        miss_anat += 1 

    
    return ok_anat, miss_anat


def log_print(message=""):
    """
    Print a message and create a list in order to save it in a .txt file.
    """
    # print(message)
    output_log.append(message)


output_log = []
def first_step_extraction(main_path, path_copy, bundles):
    """
    General first checks of the dataset and arrangement of the bundle and anat files in a new folder.
    """
    check_one_side = 0
    check_double_side = 0
    for folder in os.listdir(main_path):


        
        if os.path.isdir(os.path.join(main_path, folder)) and folder.isalnum():
            # set_to_copy.append(os.path.join(path_copy, folder))
            set_bundle_path = os.path.join(path_copy, 'bundles', folder) 
            set_anat_path = os.path.join(path_copy, 'anat', folder)
            os.makedirs(set_bundle_path, exist_ok=True)
            os.makedirs(set_anat_path, exist_ok=True)

            # output_log = []
            affine = []

            log_print(f"========== {folder} ==========")
            miss_b = 0
            miss_b_side = 0
            miss_side = 0
            n_sub = 0
            ok_anat = 0
            miss_anat = 0
            

            path_set = os.path.join(main_path, folder)
            print('\n', folder, '\n')

            for sub_folder in os.listdir(path_set):
                if 'sub' in sub_folder:
                    log_print(f"---------- {sub_folder} ----------")
                    n_sub += 1
                    os.makedirs(os.path.join(set_anat_path, sub_folder), exist_ok=True)
                    os.makedirs(os.path.join(set_bundle_path, sub_folder), exist_ok=True)

                    # print(sub_folder)

                    ok_anat, miss_anat = check_for_anat(os.path.join(path_set, sub_folder, 'anat'), ok_anat, miss_anat, os.path.join(set_anat_path, sub_folder))
                    path_sub = os.path.join(path_set, sub_folder, 'tractography')

                    sub_copy = os.path.join(set_bundle_path, sub_folder)
                    # os.makedirs(sub_copy, exist_ok=True)

                    b_check = []
                    
                    for b in bundles:
                        for file in os.listdir(path_sub):
                            # print(file)

                            if file.endswith('.trk'):
                                
                                b_name = file.split('_')[-2]
                                
                                if b_name in bundles:
                                    if b_name == b:
                                        path_file = os.path.join(path_sub, file)
                                        # if os.path.exists(os.path.join(sub_copy, file)):
                                        #     continue
                                        # else:
                                            # shutil.copy(path_file, sub_copy)


                                        
                                        # if os.path.getsize(os.path.join(path_sub, file)) > 0:
                                        #     sft = nib.streamlines.load(os.path.join(path_sub, file), lazy_load=True)
                                        #     n_streamlines = sum(1 for _ in sft.streamlines)
                                            
                                        #     if n_streamlines < 100:
                                        #         print("Under 100 streamlines", file, n_streamlines)
                                        #     # affine.append(sft.affine)
                                        # else:
                                        #     print(f"No streamlines {file}")

                                        
                

                                        b_check.append(file.split('.')[0].split('__')[-1])
                                        print(b_check)
                                        found = False
                                    
                                        for search in os.listdir(path_sub):
                                            if search != file:
                                                s_name = search.split('_')[-2]
                                                print(file, search)
                                                if s_name == b_name and file.split('_')[-1].split('.')[0] != search.split('_')[-1].split('.')[0]:
                                                    path_file_side = os.path.join(path_sub, search)
                                                    # shutil.copy(path_file_side, sub_copy)


                                                    # if os.path.getsize(os.path.join(path_sub, search)) > 0:
                                                    #     sft = nib.streamlines.load(os.path.join(path_sub, search), lazy_load=True)

                                                    #     n_streamlines = sum(1 for _ in sft.streamlines)
                                            
                                                    #     if n_streamlines < 100:
                                                    #         print("Under 100 streamlines", search, n_streamlines)
                                                    # else:
                                                    #     print(f"No streamlines {search}")



                                                    found = True
                                                    b_check.append(search.split('.')[0].split('__')[-1])
                                                    check_double_side += 1
                                                    break
                                                
                                        if not found:
                                            check_one_side += 1
                                            log_print(f"For sub {sub_folder} found ONLY this side bundle: {file.split('__')[-1]} \n")
                                        break
                    log_print("---------- BUNDLE FOR THE SUBJECT ---------- \n")
                    
                    if len(set([b.split('_')[0] for b in b_check])) != len(bundles): 
                        
                        if len(b_check) % 2 == 0:
                            log_print(f"For {sub_folder} we ONLY found the following bundles for BOTH HEMISPHERES: {set([b.split('_')[0] for b in b_check])} \n")
                            log_print(f"Missing {list(set(bundles) - set([b.split('_')[0] for b in b_check]))}")
                            miss_b += 1
                        else:
                            log_print(f"For {sub_folder} we ONLY found the following bundles: {b_check} \n")
                            log_print(f"Missing {list(set(bundles) - set([b.split('_')[0] for b in b_check]))} and also the bundle present for the single hemisphere")
                            miss_b_side += 1

                    else:
                        if len(b_check) % 2 == 0:
                            log_print(f"For {sub_folder} we found ALL the bundles of interest BOTH L and R: {bundles} \n")
                        else:
                            log_print(f"For {sub_folder} we found ALL the bundles of interest, BUT NOT ON BOTH HEMISPHERES: {b_check} \n")
                            miss_side += 1


                    log_print("-------------------------------- \n")

            log_print(f"Number of subjects in {folder}: {n_sub}\n")
            log_print(f"Subjects with missing bundles of interest: {miss_b}\n")
            if miss_b_side != 0:
                log_print(f"Subjects with missing bundles of interest considering also missing bundle in one hemisphere: {miss_b}\n")
            if miss_side != 0:
                log_print(f"Subjects with all the bundles, but with a missing bundles in one hemisphere: {miss_b}\n")



            # tot_sub =+ n_sub
            OUTPUT_PATH = f'/home/alessia/Desktop/data/TractoInferno/tractoinferno_info/{folder}'
            os.makedirs(OUTPUT_PATH, exist_ok=True)
            OUTPUT_FILE = f'{OUTPUT_PATH}/tractoinferno_dataset_{folder}.txt'
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(output_log))

            print(f"\n✅ All results have been successfully saved to '{OUTPUT_FILE}'.")


def info_sub_bundles(path):
    """
    Get info about how many subjects have a specific bundle.
    It creates 2 tables:
        1. Number of subjects for each set and for each bundle
        2. Name of subjects for each set and for each bundle
    """

    # --- Configuration ---
    
    BASE_PATH = path

    SETS = ["testset", "trainset", "validset"]

    BUNDLES = ["AF", "FAT", "ILF", "PYT", "SLF", "MdLF"]
    # ----------------------


    # List for single dataframe before merging process
    all_sets_names_list = []
    all_sets_counts_list = []
    
    for set_name in SETS:
        set_dir = os.path.join(BASE_PATH, set_name)
        
    
        if not os.path.exists(set_dir):
            print(f"Attenzione: Cartella non trovata -> {set_dir}")
            continue
            
        # Initialize dictionary for this specific set subjects
        bundle_data = {bundle: [] for bundle in BUNDLES}
        
        for subj_folder in os.listdir(set_dir):
            subj_dir = os.path.join(set_dir, subj_folder)
            
            if os.path.isdir(subj_dir):
                files = os.listdir(subj_dir)
                sub_name = subj_folder 
                
                for bundle in BUNDLES:
                    # Check bundle in both sides
                    has_left = any(f"{bundle}_L" in f for f in files)
                    has_right = any(f"{bundle}_R" in f for f in files)
                    
                    if has_left and has_right:
                        bundle_data[bundle].append(sub_name)
        
        # --- DATA PREPARATION FOR TABLE 2 ---
        df_set_names = pd.DataFrame({k: pd.Series(v) for k, v in bundle_data.items()})
        
        df_set_names.insert(0, 'Set', set_name)
        
        all_sets_names_list.append(df_set_names)
        
        # --- DATA PREPARATION FOR TABLE 1 ---
        # Creiamo un dizionario con il nome del set e il conteggio di ogni bundle
        bundle_counts = {'Set': [set_name]}
        for bundle in BUNDLES:
            bundle_counts[bundle] = [len(bundle_data[bundle])]
            
        df_set_counts = pd.DataFrame(bundle_counts)
        all_sets_counts_list.append(df_set_counts)

    # --- FINAL MERGE AND SAVE ---
    
    # TABLE 2
    if all_sets_names_list:
        df_final_names = pd.concat(all_sets_names_list, ignore_index=True)
        
        output_names_file = "/home/alessia/Desktop/data/TractoInferno_aligned/tractoinferno_info/all_sets_subjects.csv"
        df_final_names.to_csv(output_names_file, index=False)
        print(f"Creato con successo: {output_names_file}")
        print(f" -> Dimensioni tabella: {df_final_names.shape[0]} righe x {df_final_names.shape[1]} colonne\n")
    
    # TABLE 1
    if all_sets_counts_list:
        df_final_counts = pd.concat(all_sets_counts_list, ignore_index=True)
        
        output_counts_file = "/home/alessia/Desktop/data/TractoInferno_aligned/tractoinferno_info/all_sets_counts.csv"
        df_final_counts.to_csv(output_counts_file, index=False)
        print(f"Creato con successo: {output_counts_file}")
        print(f" -> Dimensioni tabella: {df_final_counts.shape[0]} righe x {df_final_counts.shape[1]} colonne")


def data_info(path_copy):
    """
    Get info about the dataset for each bundle streamlines:
    -> Set
    -> Subject
    -> Bundle
    -> Number of streamlines
    -> Mean length (mm)
    -> Standard deviation of the length (mm)
    -> Min length (mm)
    -> Max length (mm)
    """
    results = []
    for set_f in sorted(os.listdir(path_copy)):
        print(set_f)
        if not os.path.isdir(os.path.join(path_copy, set_f)):
                continue
        for sub in os.listdir(os.path.join(path_copy, set_f)):
            path_sub = os.path.join(path_copy, set_f, sub)
            

            for f in os.listdir(path_sub):
                file_path = os.path.join(path_sub, f)
                # sft = nib.streamlines.load(file_path)
                sft = load_tractogram(file_path, 'same')
                streamlines = np.asarray(sft.streamlines, dtype=object)
                n_streamlines = len(streamlines)

                mean_len = 0
                std_len = 0
                min_len = 0
                max_len = 0

                if n_streamlines > 0:
                    lengths = [np.sum(np.linalg.norm(np.diff(sl,axis=0), axis = 1)) for sl in streamlines]
                    mean_len = np.mean(lengths)
                    std_len = np.std(lengths)
                    min_len = np.min(lengths)
                    max_len = np.max(lengths)


                results.append({
                    'Dataset':set_f,
                    'Subject': sub,
                    'Bundle': f.split('.')[0].split('__')[-1],
                    'Num streamlines': n_streamlines,
                    'Mean Length mm': mean_len,
                    'Std Length mm': std_len,
                    'Min Length mm': min_len,
                    'Max Length mm': max_len,

                })

    df = pd.DataFrame(results)
    output = f"{path_copy}/streamlines_info.csv"
    df.to_csv(output, index=False)
    