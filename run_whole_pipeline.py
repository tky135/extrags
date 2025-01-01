import os, sys
sys.path.append(os.getcwd())
import json
import argparse
import datetime
from data_warehouse.file_service import DataWareHouse

DATA_WAREHOUSE_USERNAME='antbear-detection-mapping@proj'
DATA_WAREHOUSE_PASSWORD='defc0120'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='rome')
    parser.add_argument(
        '--task_id',
        default="cvg_task_id",
        help='the task id used to download data')
    parser.add_argument(
        '--workerdir',
        default="/tmp",
        help='data_path')
    parser.add_argument(
        '--upload',
        action="store_true",
        help='upload or not'
        )
    parser.add_argument(
        '--only_eval',
        action="store_true",
        help='only_eval or not'
    )
    args = parser.parse_args()
    task_id = args.task_id
    workerdir = args.workerdir
    upload = args.upload
    only_eval = args.only_eval

    current_dir = str(os.getcwd()) # "/projects/RoMe_Worker"

    if only_eval and os.path.exists(os.path.join(workerdir, "task_run.json")):
        print("eval results!")
        task_run_path = os.path.join(workerdir, "task_run.json")
        with open(task_run_path, 'r') as f:
            task_info = json.load(f)

        project_name = task_info['project_name']
        run_name = task_info['run_name']
        pth_root_path = os.path.join(workerdir, 'prod', str(task_info['task_id']), 'output')
        pth_path = os.path.join(pth_root_path, project_name, run_name, 'checkpoint_final.pth')

        cmd = f"export PYTHONPATH=$(pwd); python3 tools/eval.py --resume_from {pth_path}"
        print(cmd)
        os.system(cmd)
    else:
        # prepare workerdir
        os.system(f"mkdir -p {workerdir}")

        # import ipdb; ipdb.set_trace()
        # download task.json
        dw = DataWareHouse(print_mode=True, retry_time=5, username=DATA_WAREHOUSE_USERNAME, password=DATA_WAREHOUSE_PASSWORD, env='prod')
        task_json_path = os.path.join(workerdir, 'task.json')
        task_json_url = f'http://data-warehouse.hdmap-inner.momenta.works/api/v1/data-express/download/mesh?task_id={task_id}&file_name=task.json'
        dw.download_file(task_json_path, task_json_url)

        # get task info
        task_json_path = os.path.join(workerdir, "task.json")
        with open(task_json_path, 'r') as f:
            task_info = json.load(f)

        # download and preprocess package
        pre_process_dir = os.path.join(current_dir, "download_data")
        
        cmd = f"python3 {pre_process_dir}/download_data.py --env prod --task {task_json_path} --workdir-root {workerdir}"
        print(cmd)
        assert 0 == os.system(cmd)

        # run training
        print("start training...")
        project_name = 'test'
        run_name = 'cvg_worker_test_' + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        data_root = os.path.join(workerdir, 'prod', str(task_info['task_id']))
        scene_idx = 'package_data'
        output_root = os.path.join(workerdir, 'prod', str(task_info['task_id']), 'output')
        cmd = f"sh run.sh -p {project_name} -r {data_root} -o {output_root} -i {scene_idx} -n {run_name}"
        print(cmd)
        assert 0 == os.system(cmd)
    if upload:
        # upload data to dw
        # update task.json
        print("uploading!!!")
        task_info['project_name'] = project_name
        task_info['run_name'] = run_name
        task_json_path = os.path.join(workerdir, "task_run.json")
        with open(task_json_path, 'w') as f:
            json.dump(task_info, f, indent=4)
        post_process_dir = os.path.join(current_dir, "upload_data")
        cmd = f"python3 {post_process_dir}/upload_data.py --env prod --task {task_json_path} --workdir-root {workerdir} --skip-uploading False"
        print(cmd)
        assert 0 == os.system(cmd)
    