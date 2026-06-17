import os
import subprocess

from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm
import numpy as np
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from dipy.tracking.utils import density_map
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
        



def flip_bundle(bundle_path, folder_set):
    out_path = os.path.join(bundle_path, 'flip_bundles', folder_set)
    set_path = os.path.join(bundle_path, 'bundles', folder_set)
    os.makedirs(out_path, exist_ok=True)

    
    anat_path = os.path.join(bundle_path, 'anat', folder_set)

    csv_path = '/home/alessia.ianes/Desktop/data/TractoInferno_aligned/tractoinferno_info'
    os.makedirs(csv_path, exist_ok=True)
    results_csv = os.path.join(csv_path, f'dice_results_logicalAND_{folder_set}.csv')
    with open(results_csv, mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(['Subject', 'Flipped Bundle', 'Target Bundle', 'Dice Score'])    


        for sub in sorted(os.listdir(set_path)):
            saving_path = os.path.join(out_path, sub)
            os.makedirs(saving_path, exist_ok=True)
            sub_path = os.path.join(set_path, sub)

            for trk_file in sorted(os.listdir(sub_path)):

                if '_L' in trk_file:
                    target = trk_file.replace('_L', '_R')
                elif '_R' in trk_file:
                    target = trk_file.replace('_R', '_L')
                else:
                    continue


                file_path = os.path.join(sub_path, trk_file)
                target_path = os.path.join(sub_path, target)
                print(os.path.basename(trk_file))
                
                if not os.path.exists(target_path):
                    print(f'Target {target} not found. Skipping assessment')
                    continue
                
                ref_anat = [os.path.join(anat_path, sub, anat) for anat in os.listdir(os.path.join(anat_path, sub)) if 'warped' in anat][0]

                # print(ref_anat)

                trk = load_tractogram(file_path, ref_anat, bbox_valid_check=False)
                streamlines = trk.streamlines

                flipped_streamlines = []
                
                print("Executing flipping on x axis ...")
                for sl in streamlines:
                    sl_flipped = sl.copy()
                    sl_flipped[:, 0] = -sl_flipped[:, 0]

                    flipped_streamlines.append(sl_flipped)
                
                flipped_tractogram = StatefulTractogram(flipped_streamlines, ref_anat, Space.RASMM)

                print(f"Saving bundle {trk_file.split('__')[-1].split('.')[0]} for sub {sub}... ")

                save_tractogram(flipped_tractogram, f"{saving_path}/{trk_file.replace('.trk', '_flipped.trk')}", bbox_valid_check=False)

                print(f"Saving bundle {trk_file.split('__')[-1].split('.')[0]} for sub {sub} completed")


                    # ========== COMPUTING DICE SCORE ==========

                print(f"Loading target {target_path}")
                trk_target = load_tractogram(target_path, ref_anat, bbox_valid_check=False)
                dice_score = compute_dice_overlap(flipped_streamlines, trk_target.streamlines, ref_anat) # comment if you already have the flipped trk



                # ------ If you already have the flipped trk just uncomment the rows below ------
                # print(f"Loading flipped {saving_path}/{trk_file.replace('.trk', '_flipped.trk')}")
                # trk_flipped = load_tractogram(f"{saving_path}/{trk_file.replace('.trk', '_flipped.trk')}", ref_anat, bbox_valid_check=False)
                # dice_score = compute_dice_overlap(trk_flipped.streamlines, trk_target.streamlines, ref_anat)
                # -------------------------------------------------------------------------------





                # print(f"Sub {sub} | Flipped: {trk_file.split('__')[-1].split('.')[0]} -> Target: {target.split('__')[-1].split('.')[0]} | Dice: {dice_score:.4f}")

                writer.writerow([sub, trk_file.split('__')[-1].split('.')[0], target.split('__')[-1].split('.')[0], f'{dice_score:.4f}'])
            
                
    
def check_affine(path):
    sft = load_tractogram(path, 'same', bbox_valid_check=False)  
    print(sft.affine)


def assess_correspondence(initial_path, target_path, ref_anat, csv_file):
    # check = [os.path.join(out_path, 'bundles', folder_set, sub) for sub in initial_path]
    target_paths = [os.path.join(target_path, file) for file in sorted(os.listdir(target_path))]
    target_names = [os.path.basename(target).split('.')[0].upper() for target in target_paths]

    print(target_names)
    # target_paths = [os.path.join(target_path, f) for f in target_files]
    # print(target_paths)

    # print(target_paths[0])
    
    # print(target for target in target_files)
    # if 'PYT_L_gt.trk' in target_files:
    #     print(target_files[target_files.index('PYT_L_gt.trk')])
    # print(target for target in target_files if 'AF' in target)

    
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
                # print(file_path)
                # if bundle_name in target_names:
                #     print("ok")
                target_bundle_path = [target for target in target_paths if bundle_name in target.upper()][0]
                
                # for target in target_paths:
                #     print(target)
                #     if bundle_name in target.upper():
                #         print("ok")

                # for target in target_names:
                #     if bundle_name in target.upper():
                #         path_ref = target_paths[target_paths.index(target)]
                #         break
                # # print(.upper()])

                print(file_path)
                print(target_bundle_path)

                trk_check = load_tractogram(file_path, ref_anat, bbox_valid_check=False)
                trk_target = load_tractogram(target_bundle_path, ref_anat, bbox_valid_check=False)

                dice_score = compute_dice_overlap(trk_check.streamlines, trk_target.streamlines, ref_anat)

                writer.writerow([sub, bundle_name, target_name_out, f'{dice_score:.4f}'])






           

            

            
            # target_file = os.path.join(target_path, target)
            # if not os.path.exists(target_file):
            #     print(f'Target {target} not found. Skipping assessment')
            #     continue
            
            # check_file = os.path.join(check_path, sub, file)
            # trk_check = load_tractogram(check_file, 'same', bbox_valid_check=False)
            # trk_target = load_tractogram(target_file, 'same', bbox_valid_check=False)

            # if trk_check.affine is None or trk_target.affine is None:
            #     print(f"Affine missing for {file} or {target}. Skipping.")
            #     continue

            # if not np.allclose(trk_check.affine, trk_target.affine):
            #     print(f"Affine mismatch for {file} and {target}. Check alignment.")
            # else:
            #     print(f"Affine match for {file} and {target}.")


def dataset_reduction(path, anat, save_path):
    sft = load_tractogram(path, anat, bbox_valid_check=False)
    streamlines = np.asarray(sft.streamlines, dtype=object)

    start = perf_counter()
    prototypes = compute_dissimilarity(streamlines, hausdorff_mdf, "fft", 5000)
    end = perf_counter()
    new_tractogram = StatefulTractogram.from_sft(prototypes, sft)
    save_tractogram(new_tractogram, save_path, bbox_valid_check=False)

    print(end-start)
    with open(save_path.replace('.trk', '_TIME.txt'), 'w') as f:
        f.write(f'Time for generating prototypes {end-start}')

                



   
        





if __name__ == '__main__':

    user = 'alessia.ianes'

    folder_set = 'testset'
    sub = 'sub-1006'



    atlas_check = 'scil_merged_atlas_MNI_152_1mm'
    csv_folder = f'/home/{user}/Desktop/data/TractoInferno_aligned/tractoinferno_info/dice_scores'
    os.makedirs(csv_folder, exist_ok=True)
    csv_file = f'{csv_folder}/dice_results_densitymap_{folder_set}_{atlas_check}.csv'

    ref_path = f'/home/{user}/Desktop/data/TractoInferno/FSL_MNI152_T1_1mm.nii.gz'
    dataset_path = f'/home/{user}/Desktop/data/TractoInferno'
    out_path = f'/home/{user}/Desktop/data/TractoInferno_aligned'


    aligned_anat = f'/home/{user}/Desktop/data/TractoInferno_aligned/anat/testset/sub-1006/sub-1006__T1w_affine_warped.nii.gz'
    data_path = sorted(os.listdir(os.path.join(out_path, 'bundles', folder_set)))
    target_path = f'/home/{user}/Desktop/data/atlas_scil/bundles_of_interest/{atlas_check}'
    # target_path = f'/home/{user}/Desktop/data/atlas_scil/bundles_of_interest/{atlas_check}/SLF_R.trk'

    # assess_correspondence(data_path, target_path, aligned_anat, csv_file)


    bundle_sub_path = f'/home/{user}/Desktop/data/TractoInferno_aligned/bundles/{folder_set}/{sub}/sub-1006__AF_L_aligned_MNI.trk'
    reduction_folder = f'/home/{user}/Desktop/data/TractoInferno_aligned/reduced_bundles/{folder_set}/{sub}'
    os.makedirs(reduction_folder, exist_ok=True)
    saving_file = f'{reduction_folder}/sub-1006__AF_L_aligned_MNI_reduced.trk'
    dataset_reduction(bundle_sub_path, aligned_anat, saving_file)

   




    # ========================= SLF UNION =========================
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
