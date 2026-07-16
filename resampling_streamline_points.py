import os
import subprocess

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