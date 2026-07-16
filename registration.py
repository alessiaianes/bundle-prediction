import os
import subprocess


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