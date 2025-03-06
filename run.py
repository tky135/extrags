import datetime
import sys
import os
import fcntl
import shutil



# tmp script
TMP_SCRIPT = "nusc_run_continue_new.sh"
PRIORITY = "LOW"
# thread-safe counter
class GlobalCounter:
    def increment(self):
        if not os.path.exists("experiments"):
            os.makedirs("experiments")
        fd = os.open('experiments/.counter.txt', os.O_RDWR | os.O_CREAT)
        try:
            # Get exclusive lock
            fcntl.flock(fd, fcntl.LOCK_EX)
            # Seek to start
            os.lseek(fd, 0, os.SEEK_SET)
            # Read current value
            data = os.read(fd, 100)
            value = int(data) if data else 0
            # Write new value
            value += 1
            # Truncate and seek to start
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            # Write new value
            os.write(fd, str(value).encode())
            return value
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


# generate folder
counter = GlobalCounter()
exp_name=sys.argv[1]
scene_idx=sys.argv[2]
start_idx=sys.argv[3]
end_idx=sys.argv[4]
shift_x=sys.argv[5]

exp_dir = f"experiments/{counter.increment():04d}_{exp_name}"
os.makedirs(exp_dir)

# copy code
cpr_l = ["configs", "datasets", "magicdrive", "models", "tools", "utils"]
cpf_l = ["guidance.py", "warp.py", "utility.py", TMP_SCRIPT, "shift_x.json"]
ln_l = ["pretrained", "third_party", "BEVFormer", "MapTR"]

for cp in cpr_l:
    shutil.copytree(cp, f"{exp_dir}/{cp}")
for cp in cpf_l:
    shutil.copy(cp, f"{exp_dir}/{cp}")
for ln in ln_l:
    os.symlink(f"../../{ln}", f"{exp_dir}/{ln}")

# tmp submit job
command = f"/home/kaiyuan.tan/tmp submit job --group=ddld --machine_type=4090 --node=1 --gpu=1 --cpu=25 --memory=100 --docker_image artifactory.momenta.works/docker-momenta/hdmap-algorithm/worker-cvg:v0.0.9 --priority={PRIORITY} --work_dir={os.path.abspath(exp_dir)} --command='bash {TMP_SCRIPT} {exp_name} {scene_idx} {start_idx} {end_idx} {shift_x}'"
os.system(command)