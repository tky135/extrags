import glob
import os


for exp_num in range(1, 80):
    
    folder = glob.glob(f'experiments/{exp_num:04d}*')
    if len(folder) == 0:
        continue
    folder = folder[0]
    if os.path.exists('/train-syncdata/kaiyuan.tan/' + folder):
        continue
    os.system(f'mv {folder} /train-syncdata/kaiyuan.tan/experiments/')
    os.system(f'ln -s /train-syncdata/kaiyuan.tan/{folder} {folder}')
    print(folder)