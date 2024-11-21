import os
os.environ['CURL_CA_BUNDLE'] = ''
from huggingface_hub import hf_hub_download, snapshot_download

# Download the model from the Hub
# hf_hub_download(repo_id="ai4ce/MARS", filename="Multitraversal_2023_10_04-2024_03_08/16/16.zip", repo_type="dataset", local_dir="./MARS")
# snapshot_download(repo_id="ai4ce/MARS")



from nuscenes.nuscenes import NuScenes
location = 0

nusc = NuScenes(version='v1.0', dataroot='/home/kaiyuan.tan/drivestudio/MARS/Multitraversal_2023_10_04-2024_03_08/16/16', verbose=True)
