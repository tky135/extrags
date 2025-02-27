import json
import os
import glob

rank_fid_605, rank_fid_40, rank_fid_516 = {}, {}, {}
for exp_name in os.listdir('experiments'):
    if "_full" in exp_name:
        continue
    json_files = glob.glob(f'experiments/{exp_name}/output/exp/*/metrics/images_shift*.json')
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            fid = data["image_metrics/shift/fid"]
        scene_idx = int(json_file.split('/exp/')[1].split('/metrics')[0].split(' ')[1])


        if scene_idx == 605:
            rank_fid_605[json_file.split('/')[1]] = fid
            if fid > 83:
                os.system(f"find experiments/{exp_name} -type f -name '*.png' -delete")
                
        elif scene_idx == 40:
            rank_fid_40[json_file.split('/')[1]] = fid
            if fid > 61:
                os.system(f"find experiments/{exp_name} -type f -name '*.png' -delete")
        elif scene_idx == 516:
            rank_fid_516[json_file.split('/')[1]] = fid
            if fid > 93:
                os.system(f"find experiments/{exp_name} -type f -name '*.png' -delete")

sorted_fid = sorted(rank_fid_605.items(), key=lambda x: x[1])
for i, (k, v) in enumerate(sorted_fid):
    print(f"{i+1}. {k} - {v}")
print("=====================================")
sorted_fid = sorted(rank_fid_40.items(), key=lambda x: x[1])
for i, (k, v) in enumerate(sorted_fid):
    print(f"{i+1}. {k} - {v}")
print("=====================================")
sorted_fid = sorted(rank_fid_516.items(), key=lambda x: x[1])
for i, (k, v) in enumerate(sorted_fid):
    print(f"{i+1}. {k} - {v}")

os.system('find . -type f \( -name "*.pt" -o -name "*.ply" -o -name "*.pcd" -o -name "*.pth" \) -delete')