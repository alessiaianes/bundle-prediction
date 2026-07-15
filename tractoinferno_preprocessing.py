import os
import numpy as np
import nibabel as nib
import shutil
import pandas as pd
from dipy.io.streamline import load_tractogram





def log_print(message=""):
    # print(message)
    output_log.append(message)


def check_for_anat(path_anat, ok_anat, miss_anat, copy_dst):
    init_anat = ok_anat
    for f in os.listdir(path_anat):
        if f.endswith('.nii') or f.endswith('.nii.gz'):
            # shutil.copy(os.path.join(path_anat, f), copy_dst)
            ok_anat += 1
            break
        
    if ok_anat == init_anat:
        miss_anat += 1 

    
    return ok_anat, miss_anat


def first_step_extraction(main_path, path_copy, bundles):
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

            # same = 0
            # diff = 0
            # for a in affine:
            #     affine_pop = [aff for aff in affine if not np.allclose(a, aff, atol=1e-4)]
            #     # print(len(affine_pop))
                
            #     for b in affine_pop:
                    
            #         if np.allclose(a, b, atol=1e-4):
            #             same+=1
            #         else:
            #             # print('diff')
            #             diff+=1
            
            # print(same)
            # print(diff)


            # tot_sub =+ n_sub
            OUTPUT_PATH = f'/home/alessia/Desktop/data/TractoInferno/tractoinferno_info/{folder}'
            os.makedirs(OUTPUT_PATH, exist_ok=True)
            OUTPUT_FILE = f'{OUTPUT_PATH}/tractoinferno_dataset_{folder}.txt'
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(output_log))

            print(f"\n✅ All results have been successfully saved to '{OUTPUT_FILE}'.")
                                
                                
    # print(f"Missing anat: {miss_anat} \nAnat available: {ok_anat}")

    print(check_double_side, check_one_side)

def info_sub_bundles(path):
    import os
    import pandas as pd

    # --- CONFIGURAZIONE ---
    # Percorso base (il punto '.' indica la cartella corrente in cui si trova lo script)
    BASE_PATH = path

    # Nomi aggiornati delle cartelle dei tre set
    SETS = ["testset", "trainset", "validset"]

    # I 6 bundle di interesse
    BUNDLES = ["AF", "FAT", "ILF", "PYT", "SLF", "MdLF"]
    # ----------------------


    # Liste che conterranno i singoli dataframe di ogni set prima dell'unione finale
    all_sets_names_list = []
    all_sets_counts_list = []
    
    for set_name in SETS:
        set_dir = os.path.join(BASE_PATH, set_name)
        
        # Salta la cartella se non esiste
        if not os.path.exists(set_dir):
            print(f"Attenzione: Cartella non trovata -> {set_dir}")
            continue
            
        # Inizializza il dizionario per i soggetti di questo specifico set
        bundle_data = {bundle: [] for bundle in BUNDLES}
        
        # Scansione dei soggetti dentro il set corrente
        for subj_folder in os.listdir(set_dir):
            subj_dir = os.path.join(set_dir, subj_folder)
            
            if os.path.isdir(subj_dir):
                files = os.listdir(subj_dir)
                sub_name = subj_folder # Nome del soggetto (es. sub-1)
                
                for bundle in BUNDLES:
                    # Controllo presenza di entrambi i file (Destro e Sinistro)
                    has_left = any(f"{bundle}_L" in f for f in files)
                    has_right = any(f"{bundle}_R" in f for f in files)
                    
                    if has_left and has_right:
                        bundle_data[bundle].append(sub_name)
        
        # --- PREPARAZIONE DATI PER CSV 1 (Nomi Soggetti) ---
        # Creiamo il dataframe per il set corrente (allineando le liste con pd.Series)
        df_set_names = pd.DataFrame({k: pd.Series(v) for k, v in bundle_data.items()})
        
        # Inseriamo la colonna 'Set' in posizione 0 (la prima colonna)
        df_set_names.insert(0, 'Set', set_name)
        
        # Aggiungiamo questo dataframe alla lista globale
        all_sets_names_list.append(df_set_names)
        
        # --- PREPARAZIONE DATI PER CSV 2 (Conteggi) ---
        # Creiamo un dizionario con il nome del set e il conteggio di ogni bundle
        bundle_counts = {'Set': [set_name]}
        for bundle in BUNDLES:
            bundle_counts[bundle] = [len(bundle_data[bundle])]
            
        df_set_counts = pd.DataFrame(bundle_counts)
        all_sets_counts_list.append(df_set_counts)

    # --- UNIONE FINALE E SALVATAGGIO IN CSV ---
    
    # 1. Tabella dei Nomi (Target: ~284 x 7)
    if all_sets_names_list:
        # pd.concat unisce verticalmente i dataframe dei 3 set
        df_final_names = pd.concat(all_sets_names_list, ignore_index=True)
        
        output_names_file = "/home/alessia/Desktop/data/TractoInferno_aligned/tractoinferno_info/all_sets_subjects.csv"
        df_final_names.to_csv(output_names_file, index=False)
        print(f"Creato con successo: {output_names_file}")
        print(f" -> Dimensioni tabella: {df_final_names.shape[0]} righe x {df_final_names.shape[1]} colonne\n")
    
    # 2. Tabella dei Conteggi (Target: 3 x 7)
    if all_sets_counts_list:
        df_final_counts = pd.concat(all_sets_counts_list, ignore_index=True)
        
        output_counts_file = "/home/alessia/Desktop/data/TractoInferno_aligned/tractoinferno_info/all_sets_counts.csv"
        df_final_counts.to_csv(output_counts_file, index=False)
        print(f"Creato con successo: {output_counts_file}")
        print(f" -> Dimensioni tabella: {df_final_counts.shape[0]} righe x {df_final_counts.shape[1]} colonne")




def data_info(path_copy):
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

from tqdm import tqdm

def rearrange_dataset(origin, dest): 
    # for set_f in os.listdir(origin):
    #     for sub in tqdm(sorted(os.listdir(os.path.join(origin, set_f))), total=len(os.listdir(os.path.join(origin, set_f)))):
    #         for file in os.listdir(os.path.join(origin, set_f, sub)):
    #             dst = os.path.join(dest, file.split('__')[-1].split('_aligned')[0], set_f, sub)
    #             os.makedirs(dst, exist_ok=True)

    for sub in sorted(os.listdir(dest)):
        src = os.path.join(origin, sub)
        for file in os.listdir(src):
            shutil.copy(os.path.join(src, file), os.path.join(dest, sub))


def check_length(path):
    c = 0
    for sub in os.listdir(path):
        for file in os.listdir(os.path.join(path, sub)):
            if file.endswith('.trx'):
                c+=1

    print(c)

    # for sub in os.listdir(path):
    #     for file in os.listdir(os.path.join(path, sub)):
    #         if os.path.getsize(os.path.join(path, sub,file)) == 0:
    #             c += 1
                
    # print(c)



import subprocess

def generate_embs(config):

    bundles = ['ILF_L', 'ILF_R','MdLF_L', 'MdLF_R', 'SLF_L', 'SLF_R','PYT_L', 'PYT_R']
    sets = ['testset', 'trainset', 'validset']

    for bundle in bundles:
        for set_f in sets:
            config_file = os.path.join(config, f'config_{bundle}_{set_f}.yaml')
            cmd = ['python', '/home/alessia/project/streamline_autoencoder/save_embeddings.py', '-C', config_file, '-n', 'train_eclipse']
            subprocess.check_call(cmd)# , stdout=subprocess.DEVNULL)
            
            rearrange_dataset(f'/home/alessia/project/streamline_autoencoder/embeddings/train_eclipse', f'/home/alessia/Desktop/data/TractoInferno_rearranged/bundles_embs/{bundle}/{set_f}')
            check_length(f'/home/alessia/Desktop/data/TractoInferno_rearranged/bundles_embs/{bundle}/{set_f}')



# def create_sub_folder(path):
#     for bundle in os.listdir(path):
#         bundle_path = os.path.join(path, bundle)
#         for set_f in os.listdir(bundle_path):
#             set_path = os.path.join(bundle_path, set_f)
#             for sub in os.listdir(set_path):
#                 sub_path = os.path.join(set_path, sub)
#                 for file in os.listdir(sub_path):
#                     if not file.endswith('.trx'):
#                         continue
#                     file_path = os.path.join(sub_path, file)

#                     new_folder = '/home/alessia/Desktop/data/Tractoinferno_subjects'
#                     os.makedirs(new_folder, exist_ok=True)

#                     dst = os.path.join(new_folder, set_f, sub)
#                     os.makedirs(dst, exist_ok=True)

#                     shutil.copy(file_path, dst)
    





if __name__ ==  '__main__':
    # BUNDLE OF INTEREST
    bundles = ['AF', 'FAT', 'ILF', 'MdLF', 'SLF', 'PYT']
    path_copy = '/home/alessia/Desktop/data/TractoInferno'
    folder_interest = 'bundles' #['anat', 'bundles']
    user = 'alessia'
    bundle = 'FAT'
    side = 'R'

    main_path = '/nilab-qnap/datasets/TractoInferno/derivatives'

    output_log = []

    # src = f'/home/{user}/Desktop/data/TractoInferno_aligned/bundles'
    src = f'/home/{user}/project/streamline_autoencoder/embeddings/train_eclipse'
    dst = f'/home/{user}/Desktop/data/TractoInferno_rearranged/bundles_embs/{bundle}_{side}/trainset'

    # first_step_extraction(main_path, path_copy, bundles)
    # info_sub_bundles('/home/alessia/Desktop/data/TractoInferno_aligned/bundles')
    # data_info(os.path.join(path_copy, folder_interest))
    # rearrange_dataset(src, dst)
    # check_length(dst)

    # check_length(f'/home/{user}/Desktop/data/TractoInferno_rearranged/bundles/FAT_R/validset')


    # generate_embs(f'/home/{user}/project/streamline_autoencoder/checkpoints/train_eclipse/train_eclipse')




        