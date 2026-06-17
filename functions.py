import dipy
import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram
import json
from scipy.spatial import KDTree, cKDTree

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



def directed_hausdorff(A,B, flip=False) -> float:
    if flip:
        B = [item[::-1] for item in B]

    # mat_dist = np.zeros((len(A), len(B)))
       

    # for i, lineB in enumerate(B):
    #     treeB = KDTree(lineB)
    #     for j, lineA in enumerate(A):
    #         dist, _ = treeB.query(lineA, k=1)
    #         mat_dist[j, i] = dist.min()


    # return mat_dist

    """
    Calcola la distanza diretta vettorizzando le query spaziali su tutto il set A.
    """
    # 1. Uniamo tutte le linee di A in un unico array (N_totali, dimensioni)
    A_flat = np.vstack(A)
    
    # 2. Prepariamo gli array per il "raggruppamento" veloce dei risultati
    lens_A = np.array([len(lineA) for lineA in A])
    # Troviamo gli indici di partenza di ogni linea dentro A_flat
    # Es: lunghezze [10, 20, 5] -> indici [0, 10, 30]
    indices_A = np.insert(np.cumsum(lens_A), 0, 0)[:-1]
    
    mat_dist = np.zeros((len(A), len(B)))
    
    # 3. Pre-costruiamo gli alberi per B
    trees_B = [cKDTree(lineB) for lineB in B]
    
    # 4. Cicliamo solo su B. Il ciclo su A è sparito (completamente vettorizzato)
    for i, treeB in enumerate(trees_B):
        # Facciamo la query di TUTTI i punti di A in un colpo solo
        # n_jobs=-1 usa tutti i core della CPU per questa singola operazione
        dists, _ = treeB.query(A_flat, k=1, workers=-1)
        
        mat_dist[:, i] = np.minimum.reduceat(dists, indices_A)
            
    return mat_dist


def hausdorff_mdf(A,B) -> float:

    direct = directed_hausdorff(A, B)
    flipped = directed_hausdorff(A, B, flip=True)

    return np.minimum(direct, flipped)

        
    