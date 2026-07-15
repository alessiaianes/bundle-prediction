import dipy
from dipy.io.stateful_tractogram import StatefulTractogram
import json
from scipy.spatial import KDTree, cKDTree
import numpy as np
from scipy.spatial.distance import cdist

import os
from dipy.io.streamline import load_tractogram, save_tractogram
from pprintpp import pprint
import nibabel as nib




def initialization(bundle_path, streamline_path):
    # Loading ground truth labels
    print("="*50+'\n'+"Loading bundle labels..."+'\n'+"="*50)
    with open(bundle_path, 'r') as f:
        bundle_labels = json.load(f)

    id_to_name = {int(k) : v for k, v in bundle_labels.items()}
    labels_name = [id_to_name[i] for i in sorted(id_to_name.keys())]

    print("="*50+'\n'+"Loading streamline labels..."+'\n'+"="*50)
    streamline_labels = np.load(streamline_path)


    return labels_name, id_to_name, streamline_labels




def divide_in_bundles(gt, obj, file, names):

    for label in np.unique(gt):

        id_gt = np.where(gt == label)[0]

        sl_gt = [obj.streamlines[i] for i in id_gt]
        sft_gt = StatefulTractogram.from_sft(sl_gt, obj)


        dipy.io.streamline.save_tractogram(sft_gt, f'{file}/{names[label]}.trk', bbox_valid_check=False) #bbox_valid_check = false for HCP


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

    