import os
import json
import numpy as np
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import StatefulTractogram

from pprintpp import pprint
import nibabel as nib


if __name__ == '__main__':
	# path = '/home/alessia/project/extracting_labels/derivatives/francois/sub-14230/ses-OXF1PRI001/dwi/sub-14230_ses-OXF1PRI001_desc-brain_mask.nii.gz'
	path = '/home/alessia/Desktop/data/TractoInferno_rearranged/match_euclidean_mdf_threshold_32_new_trx'
	
	files = sorted(os.listdir(path))
	files = [item for item in files if 'trx' in item]
	
	correspondeces = dict()
	sc_correspondeces = dict()
	
	bundles = [	'AF_L', 'AF_R',
				'FAT_L', 'FAT_R', 
				'ILF_L', 'ILF_R',
				'MdLF_L', 'MdLF_R', 
				'PYT_L', 'PYT_R',
				'SLF_L', 'SLF_R'
			]

	
	all_streamlines = list()
	all_labels = list()
	
	total_bundles = len(bundles)
	# total_sc_bundles = len(sub_cortical_bundles)
	offset = 0
	
	for idx, b in enumerate(bundles):
		offset += 1
		correspondeces[idx] = b
		sc_correspondeces[idx] = b
		trk_file = [item for item in files][0]# if f'-{b}_tractography' in item][0]
		sft = load_tractogram(os.path.join(path, trk_file), reference='same')
		all_streamlines.extend(list(sft.streamlines))
		labels = [idx] * len(sft.streamlines)
		all_labels.extend(labels)
		
		if idx == total_bundles - 1:
			merged_atlas = StatefulTractogram.from_sft(all_streamlines, sft)
			save_tractogram(merged_atlas, 'scil_merged_atlas.trk')
			np.save('scil_labels.npy', np.array(all_labels))
			with open('scil_correspondences.json', 'w') as target:
				json.dump(correspondeces, target)
	
	# for idx, b in enumerate(sub_cortical_bundles):
	# 	sc_correspondeces[idx + offset] = b
	# 	trk_file = [item for item in files if f'-{b}_tractography' in item][0]
	# 	sft = load_tractogram(trk_file, reference='same')
	# 	all_streamlines.extend(list(sft.streamlines))
	# 	labels = [(idx + offset)] * len(sft.streamlines)
	# 	all_labels.extend(labels)
		
	# 	if idx == total_sc_bundles - 1:
	# 		merged_atlas = StatefulTractogram.from_sft(all_streamlines, sft)
	# 		save_tractogram(merged_atlas, 'scil_merged_atlas_with_subcortical.trk')
	# 		np.save('scil_labels_with_subcortical.npy', np.array(all_labels))
	# 		with open('scil_correspondences_with_subcortical.json', 'w') as target:
	# 			json.dump(sc_correspondeces, target)
			
	# pprint(sc_correspondeces)

