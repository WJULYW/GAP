import os
import random

import numpy as np
import scipy.io as scio
import torch
from PIL import Image
from torch.utils.data import Dataset

import utils


DATASET_LABELS = {
    "UBFC": {"bvp": True, "hr": True, "spo2": False, "rr": False},
    "BUAA": {"bvp": True, "hr": True, "spo2": False, "rr": False},
    "PURE": {"bvp": True, "hr": True, "spo2": True, "rr": False},
    "VIPL": {"bvp": False, "hr": True, "spo2": True, "rr": False},
    "V4V": {"bvp": False, "hr": True, "spo2": False, "rr": True},
    "HCW": {"bvp": True, "hr": True, "spo2": False, "rr": True},
    "HMPC-Dv1": {"bvp": True, "hr": True, "spo2": True, "rr": True},
    "VV100": {"bvp": True, "hr": True, "spo2": True, "rr": False},
}


def _safe_loadmat(path, key=None, default=None):
    if not os.path.isfile(path):
        return default
    data = scio.loadmat(path)
    if key is None:
        return data
    if key not in data:
        return default
    return data[key]


def _mat_string(value):
    arr = np.asarray(value).reshape(-1)
    if len(arr) == 0:
        return ""
    return str(arr[0])


def _window_scalar(values, start, length, default=0.0):
    if values is None:
        return np.array(default, dtype=np.float32)
    values = np.asarray(values).astype("float32").reshape(-1)
    if values.size == 0:
        return np.array(default, dtype=np.float32)
    stop = min(start + length, values.size)
    return np.array(np.nanmean(values[start:stop]), dtype=np.float32)


def _window_bvp(values, start, length):
    if values is None:
        return np.zeros(length, dtype=np.float32)
    values = np.asarray(values).astype("float32").reshape(-1)
    if values.size == 0:
        return np.zeros(length, dtype=np.float32)
    stop = min(start + length, values.size)
    out = values[start:stop]
    if out.size < length:
        out = np.pad(out, (0, length - out.size), mode="edge")
    denom = np.max(out) - np.min(out)
    if denom > 1e-6:
        out = (out - np.min(out)) / denom
    return out.astype("float32")


def _read_image(path):
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"))


def _normalize_stmap(stmap):
    stmap = stmap.astype("float32")
    out = np.empty_like(stmap)
    for c in range(stmap.shape[2]):
        values = stmap[:, :, c]
        mn = values.min(axis=1, keepdims=True)
        mx = values.max(axis=1, keepdims=True)
        out[:, :, c] = 255.0 * (values - mn) / (mx - mn + 1e-5)
    return np.clip(out, 0, 255).astype("uint8")


def _to_tensor(stmap, size=(64, 256)):
    image = Image.fromarray(stmap.astype("uint8")).resize((size[1], size[0]), Image.BILINEAR)
    arr = np.asarray(image).astype("float32") / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


def prior_augment_stmap(stmap, per_frame_permutation=True, component_scale=True):
    """Prior augmentation used by GAP.

    The component-scaling equation in the paper is typeset ambiguously. This
    implementation follows the normalized version:
    gamma * ((x - mean) / std) + gamma * mean.
    """
    aug = stmap.astype("float32").copy()
    roi_count, time_count, _ = aug.shape

    if per_frame_permutation:
        permuted = np.empty_like(aug)
        for t in range(time_count):
            permuted[:, t, :] = aug[np.random.permutation(roi_count), t, :]
        aug = permuted
    else:
        aug = aug[np.random.permutation(roi_count), :, :]

    if component_scale:
        gamma = np.random.uniform(0.8, 2.2)
        mean = aug.mean(axis=1, keepdims=True)
        std = aug.std(axis=1, keepdims=True)
        aug = gamma * ((aug - mean) / (std + 1e-6)) + gamma * mean

    return aug


class Data_DG(Dataset):
    """STMap dataset used by the original GAP MSSDG/TTPA scripts."""

    def __init__(
        self,
        root_dir,
        dataName,
        STMap,
        frames_num,
        args,
        transform=None,
        domain_label=None,
        datalist=None,
        peoplelist=None,
        output_people=False,
    ):
        self.root_dir = root_dir
        self.dataName = dataName
        self.STMap_Name = STMap
        self.frames_num = int(frames_num)
        self.datalist = sorted(os.listdir(root_dir)) if datalist is None else sorted(datalist)
        self.output_people = output_people
        self.peoplelist = list(peoplelist or [])
        self.domain_label = domain_label
        self.args = args

    def __len__(self):
        return len(self.datalist)

    def getLabel(self, nowPath, step_index):
        bvp = np.zeros(self.frames_num, dtype=np.float32)
        hr = np.array(0.0, dtype=np.float32)
        spo2 = np.array(0.0, dtype=np.float32)
        rr = np.array(0.0, dtype=np.float32)

        if self.dataName == "BUAA":
            bvp = _window_bvp(_safe_loadmat(os.path.join(nowPath, "Label/BVP.mat"), "BVP"), step_index, self.frames_num)
            hr_values = _safe_loadmat(os.path.join(nowPath, "Label/HR_256.mat"), "HR")
            if hr_values is not None:
                hr_values = np.asarray(hr_values).astype("float32").reshape(-1)
                hr = np.array(hr_values[min(int(step_index / 10), len(hr_values) - 1)], dtype=np.float32)

        elif self.dataName == "UBFC":
            bvp = _window_bvp(_safe_loadmat(os.path.join(nowPath, "Label/BVP.mat"), "BVP"), step_index, self.frames_num)
            hr = _window_scalar(_safe_loadmat(os.path.join(nowPath, "Label/HR.mat"), "HR"), step_index, self.frames_num)

        elif self.dataName == "PURE":
            bvp = _window_bvp(_safe_loadmat(os.path.join(nowPath, "Label/BVP.mat"), "BVP"), step_index, self.frames_num)
            hr = _window_scalar(_safe_loadmat(os.path.join(nowPath, "Label/HR.mat"), "HR"), step_index, self.frames_num)
            spo2 = _window_scalar(
                _safe_loadmat(os.path.join(nowPath, "Label/SPO2.mat"), "SPO2"), step_index, self.frames_num
            )

        elif self.dataName == "VIPL":
            bvp = _window_bvp(
                _safe_loadmat(os.path.join(nowPath, "Label_CSI/BVP_Filt.mat"), "BVP"),
                step_index,
                self.frames_num,
            )
            hr = _window_scalar(
                _safe_loadmat(os.path.join(nowPath, "Label_CSI/HR.mat"), "HR"), step_index, self.frames_num
            )
            spo2 = _window_scalar(
                _safe_loadmat(os.path.join(nowPath, "Label_CSI/SPO2_Filt.mat"), "SPO2"),
                step_index,
                self.frames_num,
            )

        elif self.dataName == "V4V":
            hr = _window_scalar(_safe_loadmat(os.path.join(nowPath, "Label/HR.mat"), "HR"), step_index, self.frames_num)
            rr = _window_scalar(_safe_loadmat(os.path.join(nowPath, "Label/RF.mat"), "RF"), step_index, self.frames_num)

        elif self.dataName == "HCW":
            bvp = _window_bvp(
                _safe_loadmat(os.path.join(nowPath, "Label/BVP_Filt.mat"), "BVP"), step_index, self.frames_num
            )
            hr = _window_scalar(
                _safe_loadmat(os.path.join(nowPath, "Label/HR_Filt.mat"), "HR"), step_index, self.frames_num
            )
            rr = _window_scalar(
                _safe_loadmat(os.path.join(nowPath, "Label/RF_Filt.mat"), "RF"), step_index, self.frames_num
            )

        elif self.dataName == "HMPC-Dv1":
            bvp = _window_bvp(
                _safe_loadmat(os.path.join(nowPath, "Label/bvp.mat"), "Filtered_BVP"),
                step_index,
                self.frames_num,
            )
            hr = np.array(utils.hr_cal(bvp), dtype=np.float32)
            spo2 = _window_scalar(
                _safe_loadmat(os.path.join(nowPath, "Label/SpO2.mat"), "Filtered_SpO2"),
                step_index,
                self.frames_num,
            )
            rr = np.array(utils.rr_cal(bvp), dtype=np.float32)

        elif self.dataName == "VV100":
            bvp = _window_bvp(
                _safe_loadmat(os.path.join(nowPath, "Label/BVP_Filt.mat"), "BVP"), step_index, self.frames_num
            )
            hr = _window_scalar(_safe_loadmat(os.path.join(nowPath, "Label/HR.mat"), "HR"), step_index, self.frames_num)
            spo2 = _window_scalar(_safe_loadmat(os.path.join(nowPath, "Label/SPO2.mat"), "HR"), step_index, self.frames_num)

        return hr, bvp, spo2, rr

    def _load_index(self, idx):
        index_path = os.path.join(self.root_dir, self.datalist[idx])
        item = scio.loadmat(index_path)
        now_path = _mat_string(item["Path"])
        step_index = int(np.asarray(item["Step_Index"]).reshape(-1)[0])
        return now_path, step_index

    def _make_augmented_crop(self, feature_map, step_index):
        _, max_frame, _ = feature_map.shape
        offset = 0
        if max_frame >= step_index + self.frames_num + 30:
            offset = random.randint(0, 30)
        crop = feature_map[:, step_index + offset : step_index + offset + self.frames_num, :]
        crop = prior_augment_stmap(crop)
        return crop, step_index + offset

    def __getitem__(self, idx):
        now_path, step_index = self._load_index(idx)
        person_name = os.path.basename(now_path)
        stmap_path = os.path.join(now_path, "STMap", self.STMap_Name)
        feature_map = _read_image(stmap_path)

        map_ori = feature_map[:, step_index : step_index + self.frames_num, :]
        map_aug, step_index_aug = self._make_augmented_crop(feature_map, step_index)

        hr, bvp, spo2, rr = self.getLabel(now_path, step_index)
        hr_aug, bvp_aug, spo2_aug, rr_aug = self.getLabel(now_path, step_index_aug)

        map_ori = _to_tensor(_normalize_stmap(map_ori))
        map_aug = _to_tensor(_normalize_stmap(map_aug))

        if self.output_people:
            domain_label = self.peoplelist.index(person_name) if person_name in self.peoplelist else len(self.peoplelist)
        else:
            domain_label = self.domain_label if self.domain_label is not None else 0

        return (
            map_ori,
            bvp,
            hr,
            spo2,
            rr,
            map_aug,
            bvp_aug,
            hr_aug,
            spo2_aug,
            rr_aug,
            domain_label,
        )


def spo_aug(data):
    scale_factor = torch.empty(1, device=data.device).uniform_(0.8, 2.2).item()
    mean = data.mean(dim=-1, keepdim=True)
    std = data.std(dim=-1, keepdim=True)
    return scale_factor * ((data - mean) / (std + 1e-6)) + scale_factor * mean


def getIndex(root_path, filesList, save_path, Pic_path, Step, frames_num):
    os.makedirs(save_path, exist_ok=True)
    index_paths = []
    for video_name in sorted(filesList):
        video_path = os.path.join(root_path, video_name)
        stmap_path = os.path.join(video_path, "STMap", Pic_path)
        if not os.path.isfile(stmap_path):
            continue
        feature_map = _read_image(stmap_path)
        max_frame = feature_map.shape[1]
        for step_index in range(0, max(1, max_frame - frames_num + 1), Step):
            index_name = f"{video_name}_{step_index}.mat"
            index_path = os.path.join(save_path, index_name)
            scio.savemat(index_path, {"Path": video_path, "Step_Index": step_index})
            index_paths.append(index_path)
    return index_paths
