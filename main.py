import os
from tqdm import tqdm
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed


from functions import *
from smoothing import smooth_bundles
from registration import process_sub
from matching import match_streamlines
from reduce_bundle import bundle_reduction
from resampling_streamline_points import nb_points_reduction






if __name__ == '__main__':

    user = 'alessia.ianes'


    start_total = perf_counter()

    registration = False

    if registration:
        # ===== REGISTRATION SETUP =====

        dataset_path = f'/home/{user}/Desktop/data/TractoInferno'
        folder_set = 'testset'
        out_path = f'/home/{user}/Desktop/data/TractoInferno_aligned'
        ref_path = f'/home/{user}/Desktop/data/TractoInferno/FSL_MNI152_T1_1mm.nii.gz'
        num_workers = 4
        subs = sorted(os.listdir(os.path.join(dataset_path, 'bundles', folder_set)))
        print(subs)
        

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(process_sub, dataset_path, out_path, ref_path, folder_set, s) for s in subs]

            for f in tqdm(as_completed(futures), desc='Alignment progress', total=len(futures)):
                s = f.result()


    # ========== PREPROCESSING PHASE ==========
    
    start_reducing = perf_counter()

    # ====== 1. Reduce number of streamlines
    prototype_threshold = 5000
    path_bundles = f'/home/{user}/Desktop/data/TractoInferno_rearranged/bundles'
    reduced_bundles = f'/home/{user}/Desktop/data/TractoInferno_rearranged/reduced_bundles_dynamic_parallel_sub'
    os.makedirs(reduced_bundles, exist_ok=True)

    bundle_reduction(path_bundles, reduced_bundles)

    end_reducing = perf_counter()

    print(f"Time for reducing: {end_reducing - start_reducing}")


    # ====== 2. Resampling number of points per streamline  
    start_resampling = perf_counter()

    reduced_streamlines = f'/home/{user}/Desktop/data/TractoInferno_rearranged/reduced_streamlines_NEW'
    os.makedirs(reduced_streamlines, exist_ok=True)
    points = 32

    nb_points_reduction(reduced_bundles, reduced_streamlines, points)

    end_resampling = perf_counter()

    print(f"Time for resampling {end_resampling - start_resampling}")


    # ====== 3. Flip bundles to prepare data for correspondence check
    start_flipping = perf_counter()
    flip_folder = f'/home/{user}/Desktop/data/TractoInferno_rearranged/flip_bundles_NEW'
    os.makedirs(flip_folder, exist_ok=True)
    flip_bundle(reduced_streamlines, flip_folder)
    end_flipping = perf_counter()

    print(f"Time for flipping: {end_flipping - start_flipping}")




    # ====== 4. Check correspondence
    start_matching = perf_counter()
    matching_folder = f'/home/{user}/Desktop/data/TractoInferno_rearranged/match_mdf_PARALLEL'
    match_streamlines(user, reduced_streamlines, flip_folder, matching_folder)
    end_matching = perf_counter()

    print(f"Time for matching: {end_matching - start_matching}")


    # ====== 5. Merging trx per subject
    start_merging = perf_counter()
    merged_trx_folder = f"{matching_folder}/trx"
    os.makedirs(merged_trx_folder, exist_ok=True)
    merging(merged_trx_folder, user)
    end_merging = perf_counter()

    print(f"Time for merging trx: {end_merging - start_merging}")




    # ====== 6. Smoothing bundles
    start_smoothing = perf_counter()
    smooth_bundles_folder = f'/home/{user}/Desktop/data/TractoInferno_rearranged/smooth_bundles_NEW'
    os.makedirs(smooth_bundles_folder, exist_ok=True)

    sigma = 3

    smooth_bundles(reduced_streamlines, smooth_bundles_folder, sigma)
    end_smoothing = perf_counter()

    print(f"Time for smoothing: {end_smoothing - start_smoothing}")
    

    end_total = perf_counter()


    print(f"The whole script took {end_total - start_total} seconds")







    