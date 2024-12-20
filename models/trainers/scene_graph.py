from typing import Dict
import torch
import logging

from datasets.driving_dataset import DrivingDataset
from models.trainers.base import BasicTrainer, GSModelType
from utils.misc import import_str
from utils.geometry import uniform_sample_sphere
from utility import visualize_depth_map, depth_to_rgb
from warp import warp_image

logger = logging.getLogger()

class MultiTrainer(BasicTrainer):
    def __init__(
        self,
        num_timesteps: int,
        **kwargs
    ):
        self.num_timesteps = num_timesteps
        self.log_dir = kwargs.get('log_dir', None)
        super().__init__(**kwargs)
        self.render_each_class = True
        
    def neus23dgs(self):
        if 'Ground_neus' not in self.models.keys():
            return

        points, colors = self.models['Ground_neus'].validate_mesh()
        points = torch.from_numpy(points).to(self.device)
        colors = torch.from_numpy(colors).to(self.device)
        
        ground_gs_cfg = self.gaussian_ctrl_general_cfg
        ground_gs_cfg['sh_degree'] = 3
        
        # ground_gs_model = import_str(self.model_config['Background']['type'])(class_name='Ground_gs', ctrl=ground_gs_cfg, reg=self.models['Background'].reg_cfg)
        self.models['Ground_gs'].ground_gs = True
        self.models['Ground_gs'].gaussian_2d = True
        # points[:, 2] -= self.models['Ground_neus'].cam_height
        points = (points - self.omnire_w2neus_w[:3, 3].cuda()) @ torch.linalg.inv(self.omnire_w2neus_w[:3, :3].cuda()).T 
        from_lidar = torch.zeros_like(points[:, 0]).float()
        self.models['Ground_gs'].create_from_pcd(init_means=points, init_colors=torch.ones_like(colors) * 0.5, from_lidar=from_lidar, init_opacity=0.5)
        self.gaussian_classes['Ground_gs'] = GSModelType.Ground_gs
        print("Converted neus to 2dgs")
        
        # camera model 
        import copy
        # self.color_network = copy.deepcopy(self.models['Ground_neus'].color_network)
        # self.models['Color_network'] = self.color_network
        
        # TODO enbale training
        # self.models['Color_network'].eval()
        # for param in self.models['Color_network'].parameters():
        #     param.requires_grad = False
        
        # sdf model
        self.models['Ground_gs'].sdf_network = self.models['Ground_neus'].sdf_network
        self.models['Ground_gs'].scale_mat = self.scale_mat.cuda()
        self.models['Ground_gs'].omnire_w2neus_w = self.omnire_w2neus_w.cuda()

        # TODO 测试 SDF 的normal输出
        # points_in_neus = (points  + self.omnire_w2neus_w.cuda()[:3, 3]) @ torch.linalg.inv(self.scale_mat[:3, :3]).cuda()
        # normal = self.models['SDF'].gradient_normal(points_in_neus)
        # normal = normal @ self.scale_mat[:3, :3].cuda()
        # normal = normal.squeeze()
        # normal = normal / normal.norm(dim=-1, keepdim=True)
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
        # pcd.colors = o3d.utility.Vector3dVector(colors.cpu().numpy())
        # pcd.normals = o3d.utility.Vector3dVector(normal.detach().cpu().numpy())
        
        # o3d.io.write_point_cloud("points.ply", pcd)
        
        
        # self.models['SDF'].eval()
        # for param in self.models['SDF'].parameters():
        #     param.requires_grad = False
        # del self.models['Ground']
        
    def register_normalized_timestamps(self, num_timestamps: int):
        self.normalized_timestamps = torch.linspace(0, 1, num_timestamps, device=self.device)
        
    def _init_models(self):
        # gaussian model classes
        if "Background" in self.model_config:
            self.gaussian_classes["Background"] = GSModelType.Background
        if "RigidNodes" in self.model_config:
            self.gaussian_classes["RigidNodes"] = GSModelType.RigidNodes
        if "SMPLNodes" in self.model_config:
            self.gaussian_classes["SMPLNodes"] = GSModelType.SMPLNodes
        if "DeformableNodes" in self.model_config:
            self.gaussian_classes["DeformableNodes"] = GSModelType.DeformableNodes
        if "Ground_gs" in self.model_config:
            self.gaussian_classes["Ground_gs"] = GSModelType.Ground_gs
           
        for class_name, model_cfg in self.model_config.items():
            # update model config for gaussian classes
            if class_name in self.gaussian_classes.keys():
                model_cfg = self.model_config.pop(class_name)
                self.model_config[class_name] = self.update_gaussian_cfg(model_cfg)
                
                model = import_str(model_cfg.type)(
                    **model_cfg,
                    class_name=class_name,
                    scene_scale=self.scene_radius,
                    scene_origin=self.scene_origin,
                    num_train_images=self.num_train_images,
                    device=self.device
                )
                
            elif class_name in self.misc_classes_keys:
                model = import_str(model_cfg.type)(
                    class_name=class_name,
                    **model_cfg.get('params', {}),
                    n=self.num_full_images,
                    device=self.device
                ).to(self.device)
            elif class_name == 'Ground_neus':
                model = import_str(model_cfg.type)(log_dir=self.log_dir).to(self.device)
            elif class_name == 'ExtrinsicPose' or class_name == 'ExtrinsicPose_neus':
                model = import_str(model_cfg.type)(
                    class_name=class_name,
                    **model_cfg.get('params', {}),
                    
                    
                    
                    n=7,
                    device=self.device
                ).to(self.device)
            else:
                raise Exception("Not supported class name: {}".format(class_name))
            self.models[class_name] = model
        
        
        if "rgb_decoder" in self.model_config:
            self.models['rgb_decoder'] =  RgbDecoder().to(self.device)
            
            
        if "rgb_decoder" in self.model_config:
            self.models['rgb_decoder'] =  RgbDecoder().to(self.device)
            
        logger.info(f"Initialized models: {self.models.keys()}")
        
        # register normalized timestamps
        self.register_normalized_timestamps(self.num_timesteps)
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'register_normalized_timestamps'):
                model.register_normalized_timestamps(self.normalized_timestamps)
            if hasattr(model, 'set_bbox'):
                model.set_bbox(self.aabb)
    
    def safe_init_models(
        self,
        model: torch.nn.Module,
        instance_pts_dict: Dict[str, Dict[str, torch.Tensor]]
    ) -> None:
        if len(instance_pts_dict.keys()) > 0:
            model.create_from_pcd(
                instance_pts_dict=instance_pts_dict
            )
            return False
        else:
            return True

    def init_gaussians_from_dataset(
        self,
        dataset: DrivingDataset,
    ) -> None:
        
        # Ground network initialization
        if 'Ground_neus' in self.models.keys():
            self.omnire_w2neus_w = self.models['Ground_neus'].omnire_w2neus_w.to(dataset.pixel_source.camera_data[1].cam_to_worlds.device)
            # self.cam_height = dataset.pixel_source.camera_data[0].cam_height
            self.ego_to_world = dataset.pixel_source.camera_data[0].ego_to_wrolds
            self.ego_points = dataset.pixel_source.camera_data[0].ego_to_wrolds[:, :3, 3]
            # ego points + normal 初始化
            # self.cam_0_to_neus_world = self.omnire_w2neus_w @ dataset.pixel_source.camera_data[1].cam_to_worlds
            # self.ego_points = self.cam_0_to_neus_world[:, :3, 3]
            # self.ego_points[:, 2] -= self.cam_height # MAGIC NUMBER
            self.ego_normals = self.ego_to_world[:, :3, :3] @ torch.tensor([0, 0, 1]).type(torch.float32).to(self.ego_to_world.device)
            self.models['Ground_neus'].ego_points = self.ego_points
            self.models['Ground_neus'].ego_normals = self.ego_normals
            # self.models['Ground_neus'].cam_height = self.cam_height
            self.models['Ground_neus'].max_iter = self.neus_iters
            self.models['Ground_neus'].pretrain_sdf()
            self.scale_mat = torch.from_numpy(self.models['Ground_neus'].scale_mat).to(self.device)
            # self.scale_mat = torch.linalg.inv(torch.from_numpy(self.models['Ground'].scale_mat).to(self.device))

            # lidar 初始化
            # road_lidar_pts = dataset.lidar_source.get_road_lidarpoints()
            # self.models['Ground'].pretrain_sdf_lidar(road_lidar_pts)

            # intialize NeuS
            from tqdm import trange
            t = trange(self.neus_iters, desc='pretrain road surface', leave=True)
            for step in t:

                self.models['Ground_neus'].iter_step = step

                image_infos, camera_infos = dataset.train_image_set.next(1)
                for k, v in image_infos.items():
                    if isinstance(v, torch.Tensor):
                        image_infos[k] = v.cuda(non_blocking=True)
                for k, v in camera_infos.items():
                    if isinstance(v, torch.Tensor):
                        camera_infos[k] = v.cuda(non_blocking=True)

                # forward & backward
                image_infos['is_train'] = self.training
                outputs = self.models['Ground_neus'](image_infos, camera_infos)

                if step % 500 == 0:
                    self.models['Ground_neus'].validate_image(image_infos, camera_infos, step)
                
                loss = self.models['Ground_neus'].get_loss(outputs, image_infos, camera_infos)

                t.set_description(f"pretrain road surface, loss:{loss.item()}")
                t.refresh()

                self.models['Ground_neus'].optimizer.zero_grad()
                loss.backward()
                self.models['Ground_neus'].optimizer.step()

            # # self.models['Ground_neus'].validate_mesh()

            # # convert Ground_neus to Ground_gs
            # self.neus23dgs()

            # lidar 初始化
            # road_lidar_pts = dataset.lidar_source.get_road_lidarpoints()
            # import open3d as o3d
            # # road_lidar_pts = o3d.io.read_point_cloud("lidar_loc_points_colored_road.pcd")
            # # import ipdb ; ipdb.set_trace()
            # # road_lidar_pts = torch.tensor(road_lidar_pts.points).to(self.device)
            
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(road_lidar_pts.cpu().numpy())
            # o3d.io.write_point_cloud("road_lidar.pcd", pcd)
            
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(self.ego_points.cpu().numpy())
            # o3d.io.write_point_cloud("ego.pcd", pcd)
            # # get rid of outliers
            # z_mean = road_lidar_pts[:, 2].mean()
            # z_std = road_lidar_pts[:, 2].std()
            
            # road_lidar_pts = road_lidar_pts[road_lidar_pts[:, 2] < z_mean + 2 * z_std]
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(road_lidar_pts.cpu().numpy())
            # o3d.io.write_point_cloud("road_lidar_filtered.pcd", pcd)
            
            # self.models['Ground_neus'].pretrain_sdf_lidar(road_lidar_pts)
            
            # self.models['Ground_neus'].validate_mesh()
            
            
            self.neus23dgs()
        # get instance points
        rigidnode_pts_dict, deformnode_pts_dict, smplnode_pts_dict = {}, {}, {}
        if "RigidNodes" in self.model_config:
            rigidnode_pts_dict = dataset.get_init_objects(
                cur_node_type='RigidNodes',
                **self.model_config["RigidNodes"]["init"]
            )

        if "DeformableNodes" in self.model_config:
            deformnode_pts_dict = dataset.get_init_objects(
                cur_node_type='DeformableNodes',        
                exclude_smpl="SMPLNodes" in self.model_config,
                **self.model_config["DeformableNodes"]["init"]
            )

        if "SMPLNodes" in self.model_config:
            smplnode_pts_dict = dataset.get_init_smpl_objects(
                **self.model_config["SMPLNodes"]["init"]
            )
        allnode_pts_dict = {**rigidnode_pts_dict, **deformnode_pts_dict, **smplnode_pts_dict}
        
        # NOTE: Some gaussian classes may be empty (because no points for initialization)
        #       We will delete these classes from the model_config and models
        empty_classes = [] 
        
        
        # collect models
        for class_name in self.gaussian_classes:
            model_cfg = self.model_config[class_name]
            model = self.models[class_name]
            
            empty = False
            if class_name == 'Background':                
                # ------ initialize gaussians ------
                init_cfg = model_cfg.pop('init')
                # sample points from the lidar point clouds
                if init_cfg.get("from_lidar", None) is not None:
                    sampled_pts, sampled_color, sampled_time = dataset.get_lidar_samples(
                        **init_cfg.from_lidar, device=self.device
                    )
                else:
                    sampled_pts, sampled_color, sampled_time = \
                        torch.empty(0, 3).to(self.device), torch.empty(0, 3).to(self.device), None
                from_lidar = torch.zeros(sampled_pts.shape[0], dtype=torch.float32).to(self.device)
                
                random_pts = []
                num_near_pts = init_cfg.get('near_randoms', 0)
                if num_near_pts > 0: # uniformly sample points inside the scene's sphere
                    num_near_pts *= 3 # since some invisible points will be filtered out
                    random_pts.append(uniform_sample_sphere(num_near_pts, self.device))
                num_far_pts = init_cfg.get('far_randoms', 0)
                if num_far_pts > 0: # inverse distances uniformly from (0, 1 / scene_radius)
                    num_far_pts *= 3
                    random_pts.append(uniform_sample_sphere(num_far_pts, self.device, inverse=True))
                
                if num_near_pts + num_far_pts > 0:
                    random_pts = torch.cat(random_pts, dim=0) 
                    random_pts = random_pts * self.scene_radius + self.scene_origin
                    visible_mask = dataset.check_pts_visibility(random_pts)
                    valid_pts = random_pts[visible_mask]
                    
                    sampled_pts = torch.cat([sampled_pts, valid_pts], dim=0)
                    sampled_color = torch.cat([sampled_color, torch.rand(valid_pts.shape, ).to(self.device)], dim=0)
                    from_lidar = torch.cat([from_lidar, torch.zeros(valid_pts.shape[0], dtype=torch.float32).to(self.device)], dim=0)
                # import open3d as o3d
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(sampled_pts.cpu().numpy())
                # pcd.colors = o3d.utility.Vector3dVector(sampled_color.cpu().numpy())
                # o3d.io.write_point_cloud("sampled_pts_before_filter.pcd", pcd)
                processed_init_pts = dataset.filter_pts_in_boxes(
                    seed_pts=sampled_pts,
                    seed_colors=sampled_color,
                    valid_instances_dict=allnode_pts_dict
                )
                # pcd.points = o3d.utility.Vector3dVector(processed_init_pts['pts'].cpu().numpy())
                # pcd.colors = o3d.utility.Vector3dVector(processed_init_pts['colors'].cpu().numpy())
                # o3d.io.write_point_cloud("sampled_pts_after_filter.pcd", pcd)
                
                model.create_from_pcd(
                    init_means=processed_init_pts["pts"], init_colors=processed_init_pts["colors"], from_lidar=torch.zeros_like(processed_init_pts["pts"][:, 0])
                )
                
            if class_name == 'RigidNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=rigidnode_pts_dict
                )
                
            if class_name == 'DeformableNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=deformnode_pts_dict
                )
            
            if class_name == 'SMPLNodes':
                empty = self.safe_init_models(
                    model=model,
                    instance_pts_dict=smplnode_pts_dict
                )
                
            if empty:
                empty_classes.append(class_name)
                logger.warning(f"No points for {class_name} found, will remove the model")
            else:
                logger.info(f"Initialized {class_name} gaussians")
        
        if len(empty_classes) > 0:
            for class_name in empty_classes:
                del self.models[class_name]
                del self.model_config[class_name]
                del self.gaussian_classes[class_name]
                logger.warning(f"Model for {class_name} is removed")
                
        logger.info(f"Initialized gaussians from pcd")
    
    def forward(
        self, 
        image_infos: Dict[str, torch.Tensor],
        camera_infos: Dict[str, torch.Tensor],
        novel_view: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the model

        Args:
            image_infos (Dict[str, torch.Tensor]): image and pixels information
            camera_infos (Dict[str, torch.Tensor]): camera information
                        novel_view: whether the view is novel, if True, disable the camera refinement

        Returns:
            Dict[str, torch.Tensor]: output of the model
            
        image_infos: {
            'origins': torch.Tensor, [900 / d, 1600 / d, 3]. 都是同一个origin
            'viewdirs': torch.Tensor, [900 / d, 1600 / d, 3]. 
            'direction_norm': torch.Tensor, [900 / d, 1600 / d, 1]. ???
            'pixel_coords': torch.Tensor, [900 / d, 1600 / d, 2]. normalized pixel coordinates
            'normed_time': torch.Tensor, [900 / d, 1600 / d]. normalized time. 猜测是整个bag的时间戳在0-1之间的归一化
            'img_idx': torch.Tensor, [900 / d, 1600 / d]. 
            'frame_idx': torch.Tensor, [900 / d, 1600 / d].
            'pixels': torch.Tensor, [900 / d, 1600 / d, 3]. RGB
            'sky_masks': torch.Tensor, [900 / d, 1600 / d]. 估计1代表天空
            'dynamic_masks': torch.Tensor, [900 / d, 1600 / d]. 
            'human_masks': torch.Tensor, [900 / d, 1600 / d].
            'vehicle_masks': torch.Tensor, [900 / d, 1600 / d].
            'lidar_depth_map': torch.Tensor, [900 / d, 1600 / d].
        }
        
        camera_infos: {
            'cam_id': torch.Tensor, [900 / d, 1600 / d].
            'cam_name': str.
            'camera_to_world': torch.Tensor, [4, 4]. #TODO: nuscenes相机高度从哪里来
            'height': torch.Tensor, [1]. image height
            'width': torch.Tensor, [1]. image width
            'intrinsics': torch.Tensor, [3, 3].
        }
        
        
        self.models: dict_keys(['Background', 'RigidNodes', 'DeformableNodes', 'SMPLNodes', 'Sky', 'Affine', 'CamPose'])

        self.gaussian_classes: dict_keys(['Background', 'RigidNodes', 'SMPLNodes', 'DeformableNodes'])
        """
        # set is_train
        image_infos['is_train'] = self.training
        self.models['Ground_neus'].iter_step = self.step
        # if self.step < self.neus_iters and self.training:
        #     # c2w_neus = self.models['ExtrinsicPose_neus'](camera_infos['camera_to_world'], camera_infos['cam_id'].flatten()[0])
        #     # c2w_neus = self.models['CamPose_ts_neus'](c2w_neus, image_infos['normed_time'].flatten()[0])
        #     # camera_infos['camera_to_world'] = c2w_neus
        #     outputs = {}
        #     outputs["ground"] = self.models['Ground_neus'](image_infos, camera_infos)
        #     outputs['rgb'] = torch.zeros(camera_infos['height'], camera_infos['width'], 3).to(self.device)
        #     # outputs['ground']['color_fine'] = self.affine_transformation(
        #     #         outputs['ground']['color_fine'], image_infos
        #     #     )
        #     # if self.step % 500 == 0 or image_infos['img_idx'].flatten()[0] == 81:
        #     #     self.models['Ground_neus'].validate_image(image_infos, camera_infos, self.step)
        #     return outputs
        
        # set current time or use temporal smoothing
        normed_time = image_infos["normed_time"].flatten()[0]
        self.cur_frame = torch.argmin(
            torch.abs(self.normalized_timestamps - normed_time)
        )
        
        # for evaluation
        for model in self.models.values():
            if hasattr(model, 'in_test_set'):
                model.in_test_set = self.in_test_set

        # assigne current frame to gaussian models
        for class_name in self.gaussian_classes.keys():
            model = self.models[class_name]
            if hasattr(model, 'set_cur_frame'):
                model.set_cur_frame(self.cur_frame)
        # prapare data
        if "CamPose_ts" in self.models.keys() or "ExtrinsicPose" in self.models.keys():
            processed_cam = self.process_camera(    # 如果要对pose优化，或者perturb（TODO 为什么），在这里处理
                camera_infos=camera_infos,
                image_ids=image_infos["normed_time"].flatten()[0],
                novel_view=novel_view,
                step=self.step
            )
        else:
            processed_cam = self.process_camera(    # 如果要对pose优化，或者perturb（TODO 为什么），在这里处理
                camera_infos=camera_infos,
                image_ids=image_infos["img_idx"].flatten()[0],
                novel_view=novel_view,
                step=self.step
            )
            
        # seperate camera model for neus
        # c2w_neus = self.models['ExtrinsicPose_neus'](camera_infos['camera_to_world'], camera_infos['cam_id'].flatten()[0])
        # c2w_neus = self.models['CamPose_ts_neus'](c2w_neus, image_infos['normed_time'].flatten()[0])
        # camera_infos['camera_to_world'] = c2w_neus
        camera_infos['camera_to_world'] = processed_cam.camtoworlds
        # camera_infos['camera_to_world'] = processed_cam.camtoworlds
        # processed_cam.camtoworlds = processed_cam.camtoworlds.detach()
        outputs_ground = None
        if 'Ground_gs' in self.models.keys():
            gs_ground, align_error = self.collect_gaussians(
                cam=processed_cam,
                image_ids=None,
                is_ground=True
            )
            
            # 只用路面部分训练，保持和neus一致
            outputs_ground, render_fn_ground = self.render_gaussians(
                gs=gs_ground,
                cam=processed_cam,
                near_plane=self.render_cfg.near_plane,
                far_plane=self.render_cfg.far_plane,
                render_mode="RGB+ED",
                radius_clip=self.render_cfg.get('radius_clip', 0.), 
                is_ground=True
            )
            outputs_ground['align_error'] = align_error
            
            # 2dgs只在意路面部分，在这里改变不影响3dgs
            if self.training:
                outputs_ground['rgb_gaussians'] = outputs_ground['rgb_gaussians'] * image_infos['road_masks'].unsqueeze(-1) + outputs_ground['rgb_gaussians'].detach() * (1 - image_infos['road_masks'].unsqueeze(-1))
                outputs_ground['normal'] = outputs_ground['normal']# * image_infos['road_masks'].unsqueeze(-1)
                outputs_ground['normal_from_depth'] = outputs_ground['normal_from_depth']# * image_infos['road_masks'].unsqueeze(-1)
            else:
                outputs_ground['rgb_gaussians'] = outputs_ground['rgb_gaussians']
            
        gs = self.collect_gaussians(    # 从各个gaussian model中收集gaussian，每个gaussian model都有get_gaussians接口，根据相机参数获得gaussian
            cam=processed_cam,
            image_ids=None
        ) # gs: dataclass_gs(_means, _scales, _quats, _rgbs, _opacities)
        

        # render gaussians
        # outputs: 
        #   rgb_gaussians: torch.Tensor, [900 / d, 1600 / d, 3]. 
        #   depth: torch.Tensor, [900 / d, 1600 / d, 1].
        #   opacity: torch.Tensor, [900 / d, 1600 / d, 1].
        outputs, render_fn = self.render_gaussians(
            gs=gs,
            cam=processed_cam,
            near_plane=self.render_cfg.near_plane,
            far_plane=self.render_cfg.far_plane,
            render_mode="RGB+ED",
            radius_clip=self.render_cfg.get('radius_clip', 0.),
            is_ground=False
        )
        
        # render sky
        sky_model = self.models['Sky']
        outputs["rgb_sky"] = sky_model(image_infos)
        outputs["rgb_sky_blend"] = outputs["rgb_sky"] * (1.0 - outputs["opacity"])
        
        if 'Ground_neus' in self.models.keys():
            # render ground
            ground_model = self.models['Ground_neus']
            rgb_ground = ground_model(image_infos, camera_infos)
            outputs['ground'] = rgb_ground
        # if 'Ground' in self.models.keys() and self.step <= self.neus_iters:
        #     # outputs['rgb_gaussians'] = self.models['Ground'].color_network.get_cam_encod_rgb(outputs['rgb_gaussians'], camera_infos['cam_id'])
        #     # outputs['rgb_sky'] = self.models['Ground'].color_network.get_cam_encod_rgb(outputs['rgb_sky'], camera_infos['cam_id'])
            
        #     # filter out pixels that are underground, threshold: 200m
        #     # above_ground_mask = ((outputs['depth'] < 150) | ((image_infos['viewdirs'][:, :, -1] > 0).unsqueeze(-1))).float()
            
        #     # outputs["rgb_gaussians"] = outputs["rgb_gaussians"] * above_ground_mask
        #     # outputs["opacity"] = outputs["opacity"] * above_ground_mask
        #     # affine transformation # 对每个image_idx的rgb进行一次affine transform
        #     # if False: # using sky masks
        #     #     outputs["rgb"] = self.affine_transformation(
        #     #         outputs["rgb_gaussians"] + (rgb_ground['rgb_full'] + image_infos['sky_masks'].unsqueeze(-1) * outputs["rgb_sky"]) * (1.0 - outputs["opacity"]), image_infos
        #     #         # outputs["rgb_gaussians"] + (rgb_ground['rgb_full']) * (1.0 - outputs["opacity"]) + (outputs['rgb_sky']) * torch.clip((1.0 - outputs['opacity'] - rgb_ground['opacity_full']), 0, 1), image_infos
        #     #     )
        #     # else:
        #     outputs['rgb'] = outputs["rgb_gaussians"] + (rgb_ground['rgb_full']) * (1.0 - outputs["opacity"]) + (torch.sigmoid(outputs['rgb_sky'])) * torch.clip((1.0 - outputs['opacity'] - rgb_ground['opacity_full']), 0, 1)
        #     before_affine = outputs['rgb'].detach()
        #     # if outputs['rgb'].max() > 1:
        #     #     print(outputs['rgb'].min(), outputs['rgb'].max())
        #     #     for i in range(outputs['rgb_gaussians'].shape[0]):
        #     #         for j in range(outputs['rgb_gaussians'].shape[1]):
        #     #             if outputs['rgb'][i,j].max() > 1:
        #     #                 print(i, j)
        #     #     # import ipdb ; ipdb.set_trace()
            
        #     if 'color_fine' in rgb_ground.keys():
        #         rgb_ground['color_fine'] = self.affine_transformation(
        #             rgb_ground['color_fine'], image_infos
        #         )
        #         outputs['ground'] = rgb_ground
        #     outputs["rgb"] = self.affine_transformation(
        #         # outputs["rgb_gaussians"] + (rgb_ground['rgb_full'] + image_infos['sky_masks'].unsqueeze(-1) * outputs["rgb_sky"]) * (1.0 - outputs["opacity"]), image_infos
        #         outputs['rgb'], image_infos
        #     )
            
        #     after_affine = outputs['rgb'].detach().clip(0, 1)
        #     if not self.training:
        #         outputs['rgb'] = outputs['rgb'].clip(0, 1)
        #     if self.models['Ground'].iter_step % 100 == 0 and self.training or image_infos['img_idx'].flatten()[0] == 81:
        #         with torch.no_grad():
        #             road_opacity_mean = outputs['opacity'][image_infos['road_masks'] == 1].mean()
        #             sky_opacity_mean = outputs['opacity'][image_infos['sky_masks'] == 1].mean()
        #             print(road_opacity_mean.item(), sky_opacity_mean.item())
                
        #         image_infos["before_affine"] = before_affine
        #         image_infos["after_affine"] = after_affine
        #         self.models['Ground'].validate_image(image_infos, camera_infos)
        #     if self.models['Ground'].iter_step % 10000 == 0 and self.training:
        #         self.models['Ground'].validate_mesh()
        if 'Ground_gs' in self.models.keys():
            outputs['rgb'] = outputs['rgb_gaussians'] + (outputs_ground['rgb_gaussians']) * (1.0 - outputs["opacity"]) + (outputs['rgb_sky']) * torch.clip((1.0 - outputs['opacity']) * (1 - outputs_ground['opacity']), 0, 1)
            before_affine = outputs['rgb'].detach()
            outputs["rgb"] = self.affine_transformation(
                # outputs["rgb_gaussians"] + (rgb_ground['rgb_full'] + image_infos['sky_masks'].unsqueeze(-1) * outputs["rgb_sky"]) * (1.0 - outputs["opacity"]), image_infos
                outputs['rgb'], image_infos, camera_infos
            )
            after_affine = outputs['rgb'].detach().clip(0, 1)
            outputs['outputs_ground'] = outputs_ground
            
            blended_opacity = outputs["opacity"] + outputs_ground["opacity"] * (1.0 - outputs["opacity"])
            outputs['blended_opacity'] = blended_opacity

            blended_depth = outputs["depth"] * outputs["opacity"] + outputs_ground["depth"] * (1.0 - outputs["opacity"]) * outputs_ground["opacity"]
            blended_depth = blended_depth / (blended_opacity + 1e-6)
            outputs['blended_depth'] = blended_depth
            
            image_infos['before_affine'] = before_affine
            image_infos['after_affine'] = after_affine
        else:
            raise Exception("Ground model")
        if not self.training and self.render_each_class:
            with torch.no_grad():
                for class_name in self.gaussian_classes.keys():
                    if class_name != 'Ground_gs':
                        gaussian_mask = self.pts_labels == self.gaussian_classes[class_name]
                        sep_result = render_fn(gaussian_mask)
                        sep_rgb, sep_depth, sep_opacity = sep_result['rgb_gaussians'], sep_result['depth'], sep_result['opacity']
                        outputs[class_name+"_rgb"] = self.affine_transformation(sep_rgb, image_infos, camera_infos)
                        outputs[class_name+"_opacity"] = sep_opacity
                        outputs[class_name+"_depth"] = sep_depth
                    else:
                        continue

        if not self.training or self.render_dynamic_mask:
            with torch.no_grad():
                gaussian_mask = self.pts_labels != self.gaussian_classes["Background"]
                render_result = render_fn(gaussian_mask)
                sep_rgb, sep_depth, sep_opacity = render_result['rgb_gaussians'], render_result['depth'], render_result['opacity']
                outputs["Dynamic_rgb"] = self.affine_transformation(sep_rgb, image_infos, camera_infos)
                outputs["Dynamic_opacity"] = sep_opacity
                outputs["Dynamic_depth"] = sep_depth
        if (self.step % 100 == 0) and self.training and outputs_ground is not None:
            import torchvision
            import os
            if not os.path.exists("ground_gs_rgb"):
                os.makedirs("ground_gs_rgb")
            with torch.no_grad():
                write_rgb = outputs['rgb'].detach().cpu() * 255
                
                write_rgb_ground = self.affine_transformation(outputs_ground['rgb_gaussians'], image_infos, camera_infos).detach().cpu() * 255
                write_rgb_3dgs = self.affine_transformation(outputs['rgb_gaussians'], image_infos, camera_infos).detach().cpu() * 255
                
                # write_rgb_ground = outputs_ground['rgb_gaussians'].detach().cpu() * 255
                
                write_rgb = write_rgb.permute(2, 0, 1).type(torch.uint8)
                write_gt = image_infos['pixels'].detach().cpu() * 255
                write_gt = write_gt.permute(2, 0, 1).type(torch.uint8)
                write_rgb_ground = write_rgb_ground.permute(2, 0, 1).type(torch.uint8)
                write_rgb_3dgs = write_rgb_3dgs.permute(2, 0, 1).type(torch.uint8)
                torchvision.io.write_png(torch.cat([write_rgb, write_gt, write_rgb_ground, write_rgb_3dgs], dim=1), "ground_gs_rgb/rgb_{}.png".format(self.step))
            # 监控路面和天空的透明度
            with torch.no_grad():
                if (image_infos['road_masks'] == 1).sum() > 0:
                    road_opacity_mean = outputs['opacity'][image_infos['road_masks'] == 1].mean()
                    ground_road_opacity_mean = outputs_ground['opacity'][image_infos['road_masks'] == 1].mean()
                else:
                    road_opacity_mean = torch.tensor(0)
                    ground_road_opacity_mean = torch.tensor(0)
                if (image_infos['sky_masks'] == 1).sum() > 0:
                    sky_opacity_mean = outputs['blended_opacity'][image_infos['sky_masks'] == 1].mean()
                    non_sky_opacity_mean = outputs['blended_opacity'][image_infos['sky_masks'] == 0].mean()
                else:
                    sky_opacity_mean = torch.tensor(0)
                    non_sky_opacity_mean = torch.tensor(0)
                print("3dgs road opcity:", road_opacity_mean.item(), "all sky opacity:", sky_opacity_mean.item(), "all non-sky opacity: ", non_sky_opacity_mean.item(), "2dgs road opacity:", ground_road_opacity_mean.item())
            
            # 保存road surface gs normal map
                if not os.path.exists("ground_gs_normal_map"):
                    os.makedirs("ground_gs_normal_map")
                #     shutil.rmtree("ground_gs_normal_map")
                # os.makedirs("ground_gs_normal_map")
                import torchvision
                with torch.no_grad():
                    depth_normal_vis = outputs_ground['normal_from_depth'].detach() / outputs_ground['normal_from_depth'].detach().norm(dim=-1, keepdim=True)
                    depth_normal_vis = (depth_normal_vis + 1.0) * 0.5
                    depth_normal_vis = depth_normal_vis.detach().cpu() * 255
                    normal_vis = outputs_ground['normal'].detach() / outputs_ground['normal'].detach().norm(dim=-1, keepdim=True)
                    normal_vis = (normal_vis + 1.0) * 0.5
                    normal_vis = normal_vis.detach().cpu() * 255
                    depth_normal_vis = depth_normal_vis.type(torch.uint8).permute(2, 0, 1)
                    normal_vis = normal_vis.type(torch.uint8).permute(2, 0, 1)
                    torchvision.io.write_png(depth_normal_vis, f"ground_gs_normal_map/depth_normal_{self.step}.png")
                    torchvision.io.write_png(normal_vis, f"ground_gs_normal_map/normal_{self.step}.png")
                
                image_infos["rsg"] = self.affine_transformation(outputs_ground['rgb_gaussians'], image_infos, camera_infos).detach()
                image_infos['depth_normal'] = depth_normal_vis
                image_infos['normal'] = normal_vis
            # 保存 depth 和 gt depth
            
            depth_blend_vis = depth_to_rgb(outputs['blended_depth'].squeeze(), min_val=0, max_val=100)
            depth_gt_vis = depth_to_rgb(image_infos['lidar_depth_map'].squeeze(), min_val=0, max_val=100)
            image_infos['depth_blend'] = depth_blend_vis
            image_infos['depth_gt'] = depth_gt_vis
            self.models['Ground_neus'].validate_image(image_infos, camera_infos)
            if self.step % 2000 == 0:
                self.models['Ground_neus'].validate_mesh()
            
        # import ipdb ; ipdb.set_trace()
        # target_c2w = processed_cam.camtoworlds_gt.clone()
        # if is_pseudo:
        #     target_c2w[0, 3] += 3
        # else:
        #     target_c2w[0, 3] -= 3
        # warp_img = warp_image(image_infos['pixels'], image_infos['lidar_depth_map'].squeeze(), target_c2w, processed_cam.camtoworlds_gt, camera_infos['intrinsics'])
        # warp_img *= 255
        # torchvision.io.write_png(warp_img.type(torch.uint8).cpu().permute(2, 0, 1), "warp_img.png")
        return outputs

    def compute_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
        cam_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        loss_dict = super().compute_losses(outputs, image_infos, cam_infos)
        
        return loss_dict
    
    def compute_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        image_infos: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        metric_dict = super().compute_metrics(outputs, image_infos)
        
        return metric_dict