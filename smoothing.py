import os
import subprocess

def smooth_bundles(in_path, out_path, sigma):

    for bundle in sorted(os.listdir(in_path)):
        for set_f in sorted(os.listdir(os.path.join(in_path, bundle))):
            for sub in sorted(os.listdir(os.path.join(in_path, bundle, set_f))):
                out_folder = os.path.join(out_path, bundle, set_f, sub)
                os.makedirs(out_folder, exist_ok=True)
                for file in os.listdir(os.path.join(in_path, bundle, set_f, sub)):
                    file_path = os.path.join(in_path, bundle, set_f, sub, file)

                    file_out_path = os.path.join(out_folder, file.replace('_32_points.trk', '_smooth.trk'))

                    print(f"Smoothing bundle {bundle} for sub {sub}...")

                    cmd = ['scil_tractogram_smooth', '--gaussian', str(sigma), '-f', file_path, file_out_path]

                    subprocess.check_call(cmd, stdout=subprocess.DEVNULL)
                    