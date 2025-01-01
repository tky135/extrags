from typing import List, Tuple
import torch
import torch.multiprocessing as mp
import random
from .pixel_source import ScenePixelSource
import torch.multiprocessing as mp
from threading import Lock
import random
from typing import Tuple, Dict
from torch.utils.data import DataLoader
import heapq

class SplitWrapper(torch.utils.data.IterableDataset):
    def __init__(
        self,
        datasource: ScenePixelSource,
        split_indices: List[int] = None,
        split: str = "train",
        camera_downscale: float = 1.0
    ):
        super().__init__()
        self.datasource = datasource
        self.split_indices = split_indices
        self.split = split
        self.modes2idx = {"random": 0, "sequential": 1}
        self._shared_index = mp.Value('i', 0)
        self._camera_downscale = mp.Value('d', camera_downscale)
        self._camera_downscale_lock = mp.Lock()
        self._mode = mp.Value('i', 0)
        self._mode_lock = mp.Lock()
        self.available_indices = list(range(len(self.split_indices)))
        

    @property
    def camera_downscale(self):
        with self._camera_downscale_lock:
            return self._camera_downscale.value
    @property
    def mode(self):
        with self._mode_lock:
            return self._mode.value
    @mode.setter
    def mode(self, value):
        with self._mode_lock:
            self._mode.value = self.modes2idx[value]
    @camera_downscale.setter
    def camera_downscale(self, value):
        with self._camera_downscale_lock:
            self._camera_downscale.value = value
        print(f"Camera downscale set to {value}")

    # def get_image(self, idx, camera_downscale, lane_shift: bool = False) -> dict:
    #     downscale_factor = 1 / camera_downscale * self.datasource.downscale_factor
    #     self.datasource.update_downscale_factor(downscale_factor)
    #     image_infos, cam_infos = self.datasource.get_image(self.split_indices[idx])
    #     if lane_shift:
    #         cam_infos['camera_to_world'][0, 3] -= 3
    #     self.datasource.reset_downscale_factor()
    #     return image_infos, cam_infos
    def get_iterator(self, num_workers: int = 4, prefetch_factor: int = 2):
        if self._mode.value == 0:
            return iter(DataLoader(
            dataset=self,
            batch_size=1,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            persistent_workers=False,
            shuffle=False
        ))
        elif self._mode.value == 1:
            # reset test iterator
            self.reset_test_iterator()
            # For test mode, we need to maintain order
            raw_loader = DataLoader(
                dataset=self,
                batch_size=1,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=True,
                persistent_workers=False,
                shuffle=False
            )

            class OrderedIterator:
                def __init__(self, loader):
                    self.loader = loader
                    self.buffer = []
                    self.next_idx = 0
                    self._exhausted = False
                    
                def __iter__(self):
                    return self
                    
                def __next__(self):
                    while True:
                        # Try to get item from buffer first
                        while self.buffer and self.buffer[0][0] == self.next_idx:
                            self.next_idx += 1
                            _, item = heapq.heappop(self.buffer)
                            return item
                        
                        # If buffer is empty and loader is exhausted, we're done
                        if self._exhausted and not self.buffer:
                            raise StopIteration
                        
                        # Get next item from loader
                        try:
                            image_infos, cam_infos = next(self.loader)
                            idx = image_infos['_sequence_idx'].item()
                            del image_infos['_sequence_idx']  # Remove temporary index
                            
                            if idx == self.next_idx:
                                self.next_idx += 1
                                return image_infos, cam_infos
                            else:
                                heapq.heappush(self.buffer, (idx, (image_infos, cam_infos)))
                                
                        except StopIteration:
                            self._exhausted = True
                            if not self.buffer:
                                raise StopIteration
            return OrderedIterator(iter(raw_loader))
        else:
            raise Exception("Invalid mode")
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if self._mode.value == 0:
            if worker_info is not None:
                worker_id = worker_info.id
                num_workers = worker_info.num_workers
                seed = torch.initial_seed()
                torch.manual_seed(seed + worker_id)
                random.seed(seed + worker_id + num_workers)
            else:
                random.seed(torch.initial_seed())
            return self._generate_random_samples()
        elif self._mode.value == 1:
            return self._generate_sequential_samples()
        else:
            import ipdb ; ipdb.set_trace()
            raise Exception("Invalid mode")

    def _generate_random_samples(self):
        while True:
            img_idx = self.datasource.propose_training_image(
                candidate_indices=self.split_indices
            )
            
            current_downscale = self.camera_downscale
            
            downscale_factor = 1 / current_downscale * self.datasource.downscale_factor
            self.datasource.update_downscale_factor(downscale_factor)
            image_infos, cam_infos = self.datasource.get_image(img_idx)
            self.datasource.reset_downscale_factor()
            
            yield image_infos, cam_infos

    def _generate_sequential_samples(self):
        while True:
            with self._shared_index.get_lock():
                current_idx = self._shared_index.value
                if current_idx >= len(self.split_indices):
                    break
                self._shared_index.value = current_idx + 1
            if current_idx >= len(self.available_indices):
                break
            current_downscale = self.camera_downscale
            
            downscale_factor = 1 / current_downscale * self.datasource.downscale_factor
            self.datasource.update_downscale_factor(downscale_factor)
            image_infos, cam_infos = self.datasource.get_image(self.split_indices[self.available_indices[current_idx]])
            self.datasource.reset_downscale_factor()
            
            # Add sequence index to outputs
            image_infos['_sequence_idx'] = current_idx
            
            yield image_infos, cam_infos

    def reset_test_iterator(self):
        with self._shared_index.get_lock():
            self._shared_index.value = 0