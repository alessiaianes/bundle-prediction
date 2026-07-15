import numpy as np
from scipy.optimize import linear_sum_assignment
from functions import pairwise_mdf

import os

from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram


def load_bundle(trk_path: str):
    """
    Load a .trk and return a list of streamlines in rasmm space.
    """
    sft = load_tractogram(trk_path, "same", bbox_valid_check=False)
    sft.to_rasmm()
    return list(sft.streamlines), sft


def save_bundle(streamlines, reference_sft, out_path: str):
    """
    Save a subset of streamlines as .trk, using the same affine as the reference tractogram.
    """
    out_sft = StatefulTractogram.from_sft(
        streamlines,
        reference_sft,
        # space=Space.RASMM,
    )
    save_tractogram(out_sft, out_path, bbox_valid_check=False)
    print(f"Salvato: {out_path}  ({len(streamlines)} streamlines)")





def save_results(rows, cols, match, folder, filename):
    res = []
    for i in range(len(match)):
        res.append((rows[i], cols[i], match[i]))

    print("Saving results...")
    os.makedirs(folder, exist_ok=True)
    np.save(f"{folder}/{filename}.npy", res)


def match_streamlines(user, path_sl, path_flip):
    from time import perf_counter

    start = perf_counter()
    mdf_threshold = 10.0
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




                    # ========== MATCHING 1-1 RESAMPLING TO 16 POINTS ==========
                    print(f"Loading bundles {bundle} and flipped {bundle_opp} for sub {sub}...")
                    sl_a, sft_a = load_bundle(path_file)
                    sl_b, sft_b = load_bundle(path_opp)


                    print("Computing distances using pairwise mdf...")
                    distances = pairwise_mdf(np.array(sl_a), np.array(sl_b))
                 



                    print("Checking 1-1 match....")
                    row_ind, col_ind = linear_sum_assignment(distances)
                    matched_pairs = distances[row_ind, col_ind]

                    if mdf_threshold is not None:
                        valid = matched_pairs <= mdf_threshold
                        n_valid = valid.sum()
                        print(f"  Match under threshold {mdf_threshold} mm: {n_valid}/{len(col_ind)}")
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

           
                    matched_streamlines = np.array(sl_b)[valid]
                    out_sft = StatefulTractogram.from_sft(
                        np.array(sl_a)[valid],
                        sft_a,
                    )
                   
                    out_sl = np.array([np.array(sl) for sl in matched_streamlines])
                    compress = out_sl.reshape(out_sl.shape[0], -1)

                    # # to get original array
                    # # sft.data_per_streamline['match']
                    # # original = sft.reshape[-1, 32, 3]
                  
                    out_sft.data_per_streamline['match'] = compress
                    trx_folder = f"/home/{user}/Desktop/data/TractoInferno_rearranged/{match_folder}_trx/{set_f}/{sub}"
                    os.makedirs(trx_folder, exist_ok=True)     
                    save_tractogram(out_sft, f"{trx_folder}/{bundle}_matched.trx", bbox_valid_check=False)

                    match_trk = f"/home/{user}/Desktop/data/TractoInferno_rearranged/{match_folder}_trk/{set_f}/{sub}"
                    os.makedirs(match_trk, exist_ok=True)
                    save_bundle(matched_streamlines, sft_b, f"{match_trk}/{bundle_opp}_to_{bundle}_32_points.trk")
                    

                    

    end = perf_counter()
    print(f"Total time: {end - start:.2f} seconds")

