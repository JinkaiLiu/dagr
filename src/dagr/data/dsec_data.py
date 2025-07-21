from pathlib import Path
from typing import Optional, Callable

from torch_geometric.data import Dataset

import numpy as np
import cv2

import torch
from functools import lru_cache

from dsec_det.dataset import DSECDet

from dsec_det.io import yaml_file_to_dict
from dagr.data.dsec_utils import filter_tracks, crop_tracks, rescale_tracks, compute_class_mapping, map_classes, filter_small_bboxes
from dsec_det.directory import BaseDirectory
from dagr.data.augment import init_transforms
from dagr.data.utils import to_data

from dagr.visualization.bbox_viz import draw_bbox_on_img
from dagr.visualization.event_viz import draw_events_on_image


def tracks_to_array(tracks):
    return np.stack([tracks['x'], tracks['y'], tracks['w'], tracks['h'], tracks['class_id']], axis=1)



def interpolate_tracks(detections_0, detections_1, t):
    assert len(detections_1) == len(detections_0)
    if len(detections_0) == 0:
        return detections_1

    t0 = detections_0['t'][0]
    t1 = detections_1['t'][0]

    assert t0 < t1

    # need to sort detections
    detections_0 = detections_0[detections_0['track_id'].argsort()]
    detections_1 = detections_1[detections_1['track_id'].argsort()]

    r = ( t - t0 ) / ( t1 - t0 )
    detections_out = detections_0.copy()
    for k in 'xywh':
        detections_out[k] = detections_0[k] * (1 - r) + detections_1[k] * r

    return detections_out

class EventDirectory(BaseDirectory):
    @property
    @lru_cache
    def event_file(self):
        return self.root / "left/events_2x.h5"


class DSEC(Dataset):
    MAPPING = dict(pedestrian="pedestrian", rider=None, car="car", bus="car", truck="car", bicycle=None,
                   motorcycle=None, train=None)
    def __init__(self,
                 root: Path,
                 split: str,
                 transform: Optional[Callable]=None,
                 debug=False,
                 min_bbox_diag=0,
                 min_bbox_height=0,
                 scale=2,
                 cropped_height=430,
                 only_perfect_tracks=False,
                 demo=False,
                 no_eval=False):

        Dataset.__init__(self)

        split_config = None
        if not demo:
            split_config = yaml_file_to_dict(Path(__file__).parent / "dsec_split.yaml")
            assert split in split_config.keys(), f"'{split}' not in {list(split_config.keys())}"

        self.dataset = DSECDet(root=root, split=split, sync="back", debug=debug, split_config=split_config)

        # Output found directories
        print(f"[DEBUG] Found directories: {list(self.dataset.directories.keys())}")

        valid_sequences = []
        for name, directory in self.dataset.directories.items():
            try:
                # Set event directory
                directory.events = EventDirectory(directory.events.root)

                images_valid = False
                events_valid = False

                # Check if images might be valid
                if hasattr(directory, 'images'):
                    # Try checking some possible image attributes
                    for attr_name in ['timestamps', 'image_files', 'filenames', 'paths']:
                        if hasattr(directory.images, attr_name):
                            attr = getattr(directory.images, attr_name)
                            if attr is not None and hasattr(attr, '__len__') and len(attr) > 0:
                                images_valid = True
                                print(f"[DEBUG] Directory {name} has valid images via '{attr_name}' attribute")
                                break

                # Check if event file exists
                try:
                    events_valid = directory.events.event_file.exists()
                except Exception as e:
                    print(f"[WARNING] Error checking event file for {name}: {e}")

                # Keep this sequence if images and events are valid, or at least images are valid
                if images_valid:
                    valid_sequences.append(name)
                    print(f"[INFO] Directory {name} appears valid. Events: {events_valid}")
                else:
                    print(f"[WARNING] Directory {name} missing necessary files. Images valid: {images_valid}, Events valid: {events_valid}")

            except Exception as e:
                print(f"[ERROR] Error processing directory {name}: {e}")

        # If there are invalid sequences, optionally filter them out
        if len(valid_sequences) < len(self.dataset.directories) and len(valid_sequences) > 0:
            print(f"[INFO] Found {len(valid_sequences)} valid sequences out of {len(self.dataset.directories)}")
            self.dataset.directories = {name: self.dataset.directories[name] for name in valid_sequences}
            self.dataset.subsequence_directories = [d for d in self.dataset.subsequence_directories if d.name in valid_sequences]

        self.debug = debug

        #self.dataset = DSECDet(root=root, split=split, sync="back", debug=debug)
        self.split = split
        self.root = root

        # Automatically detect and use 'transformed_images' if it exists
       # transformed_root = root / split / "transformed_images"
      #  if transformed_root.exists():
     #       print(f"[DEBUG] Using transformed_images folder: {transformed_root}")
    #        self.root = transformed_root
   #     else:
  #          print(f"[DEBUG] Using standard folder: {root / split}")
 #           self.root = root / split

        # Initialize DSECDet with the adjusted root
#        self.dataset = DSECDet(root=self.root, split=split, sync="back", debug=debug)

        print(f"[DEBUG] Initialized DSEC with root: {self.root}")
        print(f"[DEBUG] Available directories: {list(self.dataset.directories.keys())}")

        self.scale = scale
        self.width = self.dataset.width // scale
        self.height = cropped_height // scale
        self.classes = ("car", "pedestrian")
        self.time_window = 1000000
        self.min_bbox_height = min_bbox_height
        self.min_bbox_diag = min_bbox_diag
        self.num_us = -1

        self.class_remapping = compute_class_mapping(self.classes, self.dataset.classes, self.MAPPING)

        if transform is not None and hasattr(transform, "transforms"):
            init_transforms(transform.transforms, self.height, self.width)

        self.transform = transform
        self.no_eval = no_eval

        if self.no_eval:
            only_perfect_tracks = False

        self.image_index_pairs, self.track_masks = filter_tracks(
            dataset=self.dataset,
            image_width=self.width,
            image_height=self.height,
            class_remapping=self.class_remapping,
            min_bbox_height=min_bbox_diag,
            min_bbox_diag=min_bbox_diag,
            only_perfect_tracks=only_perfect_tracks,
            scale=scale
        )

        if debug:
            print(f"\n[DEBUG] Initialized DSEC with root: {root}")
            print(f"[DEBUG] Split: {split}")
            print(f"[DEBUG] Checking directory structure in root: {root / split}")

            if not (root / split).exists():
                print(f"[ERROR] Split directory does not exist: {root / split}")
                raise FileNotFoundError(f"Split directory not found: {root / split}")

#            for sequence in (root / split / 'transformed_images').iterdir():
            for sequence in (root / split).iterdir():
                if sequence.is_dir():
                    print(f"\n[DEBUG] Sequence: {sequence.name}")

                    # check Images folder
                    images_path = sequence / "images" / "left" / "rectified"
                    if images_path.exists():
                        images_files = list(images_path.rglob("*.png"))
                        print(f"  -> Images ({len(images_files)} files):")
                        for img in images_files[:5]:
                            print(f"    - {img.name}")
                        if len(images_files) > 5:
                            print(f"    ... and {len(images_files) - 5} more")
                    else:
                        print(f"  -> Images: Not found")

                    # check Object Detections folder
                    object_detections_path = sequence / "object_detections" / "left"
                    if object_detections_path.exists():
                        detections_files = list(object_detections_path.rglob("*.npy"))
                        print(f"  -> Object Detections ({len(detections_files)} files):")
                        for det in detections_files:
                            print(f"    - {det.name}")
                    else:
                        print(f"  -> Object Detections: Not found")

                    # check Events folder
                    events_path = sequence / "events" / "left"
                    if events_path.exists():
                        events_files = list(events_path.rglob("*.h5"))
                        print(f"  -> Events ({len(events_files)} files):")
                        for evt in events_files:
                            print(f"    - {evt.name}")
                    else:
                        print(f"  -> Events: Not found")

                    # check Timestamps.txt
                    timestamps_path = sequence / "images" / "timestamps.txt"
                    if timestamps_path.exists():
                        print(f"  -> Timestamps: Exists")
                    else:
                        print(f"  -> Timestamps: Not found")
        split_config = None
        if not demo:
            split_config = yaml_file_to_dict(Path(__file__).parent / "dsec_split.yaml")
            assert split in split_config.keys(), f"'{split}' not in {list(split_config.keys())}"

        self.dataset = DSECDet(root=root, split=split, sync="back", debug=debug, split_config=split_config)

        print(f"[DEBUG] Dataset root: {root}")
        print(f"[DEBUG] Split: {split}")
        print(f"[DEBUG] Available directories: {list(self.dataset.directories.keys())}")

        for directory in self.dataset.directories.values():
            directory.events = EventDirectory(directory.events.root)

        self.scale = scale
        self.width = self.dataset.width // scale
        self.height = cropped_height // scale
        self.classes = ("car", "pedestrian")
        self.time_window = 1000000
        self.min_bbox_height = min_bbox_height
        self.min_bbox_diag = min_bbox_diag
        self.debug = debug
        self.num_us = -1

        self.class_remapping = compute_class_mapping(self.classes, self.dataset.classes, self.MAPPING)

        if transform is not None and hasattr(transform, "transforms"):
            init_transforms(transform.transforms, self.height, self.width)

        self.transform = transform
        self.no_eval = no_eval

        if self.no_eval:
            only_perfect_tracks = False

        self.image_index_pairs, self.track_masks = filter_tracks(dataset=self.dataset, image_width=self.width,
                                                                 image_height=self.height,
                                                                 class_remapping=self.class_remapping,
                                                                 min_bbox_height=min_bbox_height,
                                                                 min_bbox_diag=min_bbox_diag,
                                                                 only_perfect_tracks=only_perfect_tracks,
                                                                 scale=scale)

    def set_num_us(self, num_us):
        self.num_us = num_us

    def visualize_debug(self, index):
        data = self.__getitem__(index)
        image = data.image[0].permute(1,2,0).numpy()
        p = data.x[:,0].numpy()
        x, y = data.pos.t().numpy()
        b_x, b_y, b_w, b_h, b_c = data.bbox.t().numpy()

        image = draw_events_on_image(image, x, y, p)
        image = draw_bbox_on_img(image, b_x, b_y, b_w, b_h,
                                 b_c, np.ones_like(b_c), conf=0.3, nms=0.65)

        cv2.imshow(f"Debug {index}", image)
        cv2.waitKey(0)


    def __len__(self):
        return sum(len(d) for d in self.image_index_pairs.values())

    def preprocess_detections(self, detections):
        detections = rescale_tracks(detections, self.scale)
        detections = crop_tracks(detections, self.width, self.height)
        detections['class_id'], _ = map_classes(detections['class_id'], self.class_remapping)
        return detections

    def preprocess_events(self, events):
        mask = events['y'] < self.height
        events = {k: v[mask] for k, v in events.items()}
        if len(events['t']) > 0:
            events['t'] = self.time_window + events['t'] - events['t'][-1]
        events['p'] = 2 * events['p'].reshape((-1,1)).astype("int8") - 1
        return events

    def preprocess_image(self, image):
        image = image[:self.scale * self.height]
        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_CUBIC)
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = image.unsqueeze(0)
        return image

    def __getitem__(self, idx):
        #print(f"[DEBUG] __getitem__ called with index {idx}")
        dataset, image_index_pairs, track_masks, idx = self.rel_index(idx)
        image_index_0, image_index_1 = image_index_pairs[idx]
        image_ts_0, image_ts_1 = dataset.images.timestamps[[image_index_0, image_index_1]]

        detections_0 = self.dataset.get_tracks(image_index_0, mask=track_masks, directory_name=dataset.root.name)
        detections_1 = self.dataset.get_tracks(image_index_1, mask=track_masks, directory_name=dataset.root.name)

        detections_0 = self.preprocess_detections(detections_0)
        detections_1 = self.preprocess_detections(detections_1)

        image_0 = self.dataset.get_image(image_index_0, directory_name=dataset.root.name)
        image_0 = self.preprocess_image(image_0)

       # detections_0, detections_1 = None, None
       # for split in ["train", "test"]:
       #     track_dir_0 = dataset.root / split / sequence / "object_detections" / "left" / "tracks.npy"
       #     track_dir_1 = dataset.root / split / "object_detections" / "left" / "tracks.npy"

       #     print(f"[DEBUG] Checking label paths: {track_dir_0}, {track_dir_1}")

       #     if track_dir_0.exists():
        #        print(f"[DEBUG] Loading labels from {track_dir_0}")
       #         detections_0 = np.load(str(track_dir_0))
       #         detections_1 = np.load(str(track_dir_0))
       #         break

       #     if track_dir_1.exists():
       #         print(f"[DEBUG] Loading labels from {track_dir_1}")
       #         detections_0 = np.load(str(track_dir_1))
       #         detections_1 = np.load(str(track_dir_1))
       #         break

        if detections_0 is None or detections_1 is None:
            raise FileNotFoundError("Track files not found in train or test directories.")

      #  detections_0 = self.preprocess_detections(detections_0)
      #  detections_1 = self.preprocess_detections(detections_1)


      #  image_0 = None
      #  for split in ["train", "test"]:
      #      image_path_0 = dataset.root / split / sequence / "images" / "left" / "rectified" / f"{image_index_0:06d}.png"
      #      image_path_1 = dataset.root / split / "images" / "left" / "rectified" / f"{image_index_0:06d}.png"

      #      print(f"[DEBUG] Checking image paths: {image_path_0}, {image_path_1}")

      #      if image_path_0.exists():
      #          print(f"[DEBUG] Loading image from {image_path_0}")
      #          image_0 = cv2.imread(str(image_path_0))
      #          break

      #      if image_path_1.exists():
      #          print(f"[DEBUG] Loading image from {image_path_1}")
      #          image_0 = cv2.imread(str(image_path_1))
      #          break

        if image_0 is None:
            raise FileNotFoundError("Image file not found in train or test directories.")

      #  image_0 = self.preprocess_image(image_0)

        events = self.dataset.get_events(image_index_0, directory_name=dataset.root.name)

        if self.num_us >= 0:
           image_ts_1 = image_ts_0 + self.num_us
           events = {k: v[events['t'] < image_ts_1] for k, v in events.items()}
           if not self.no_eval:
            	detections_1 = interpolate_tracks(detections_0, detections_1, image_ts_1)


        # here, the timestamp of the events is no longer absolute
        events = self.preprocess_events(events)

        # convert to torch geometric data
        data = to_data(**events, bbox=tracks_to_array(detections_1), bbox0=tracks_to_array(detections_0), t0=image_ts_0, t1=image_ts_1,
                       width=self.width, height=self.height, time_window=self.time_window,
                       image=image_0, sequence=str(dataset.root.name))

        if self.transform is not None:
            data = self.transform(data)

        # remove bboxes if they have 0 width or height
        mask = filter_small_bboxes(data.bbox[:, 2], data.bbox[:, 3], self.min_bbox_height, self.min_bbox_diag)
        data.bbox = data.bbox[mask]
        mask = filter_small_bboxes(data.bbox0[:, 2], data.bbox0[:, 3], self.min_bbox_height, self.min_bbox_diag)
        data.bbox0 = data.bbox0[mask]

        return data

    def rel_index(self, idx):
        for folder in self.dataset.subsequence_directories:
            name = folder.name
            image_index_pairs = self.image_index_pairs[name]
            directory = self.dataset.directories[name]
            track_mask = self.track_masks[name]
            if idx < len(image_index_pairs):
                return directory, image_index_pairs, track_mask, idx
            idx -= len(image_index_pairs)
        raise IndexError
