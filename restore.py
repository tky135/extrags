
import glob
import os
import sys

exp_num = int(sys.argv[1])
exp_folder = glob.glob(f'experiments/{exp_num:04d}*')[0]


for folder in ['configs', 'datasets', 'models', 'tools']:
    os.system(f'rm -rf {folder}')
    os.system(f'cp -r {exp_folder}/{folder} .')

for file in ['guidance.py']:
    os.system(f'rm {file}')
    os.system(f'cp {exp_folder}/{file} .')
