import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from scipy.fft import fft
from scipy.signal import butter, filtfilt


def slice_bvp(input, slice_lens=11):
    if input.dim() == 3:
        input = input.squeeze(1)
    slices = input.unfold(dimension=-1, size=slice_lens, step=1)
    return slices


def cal_cos_similarity_self(tensor):
    norm_tensor = F.normalize(tensor, p=2, dim=2)
    return torch.matmul(norm_tensor, norm_tensor.transpose(1, 2))


def hr_fft(sig, fs=30, harmonics_removal=True):
    sig = np.asarray(sig).reshape(-1)
    sig = sig * signal.windows.hann(sig.shape[0])
    sig_f = np.abs(fft(sig))
    low_idx = np.round(0.6 / fs * sig.shape[0]).astype("int")
    high_idx = np.round(3 / fs * sig.shape[0]).astype("int")
    sig_f_original = sig_f.copy()
    sig_f[:low_idx] = 0
    sig_f[high_idx:] = 0
    peak_idx, _ = signal.find_peaks(sig_f)
    if len(peak_idx) == 0:
        return 0.0, sig_f_original, None
    sort_idx = np.argsort(sig_f[peak_idx])[::-1]
    peak_idx1 = peak_idx[sort_idx[0]]
    hr1 = peak_idx1 / sig.shape[0] * fs * 60
    if harmonics_removal and len(sort_idx) > 1:
        peak_idx2 = peak_idx[sort_idx[1]]
        hr2 = peak_idx2 / sig.shape[0] * fs * 60
        hr = hr2 if np.abs(hr1 - 2 * hr2) < 10 else hr1
    else:
        hr = hr1
    x_hr = np.arange(len(sig)) / len(sig) * fs * 60
    return hr, sig_f_original, x_hr


def hr_cal(bvp_signal, fps=30):
    hr, _, _ = hr_fft(bvp_signal, fs=fps, harmonics_removal=True)
    return hr


def rr_cal(bvp_signal, fs=30):
    bvp_signal = np.asarray(bvp_signal).reshape(-1)
    if len(bvp_signal) < 8:
        return 0.0
    nyquist = 0.5 * fs
    low = 0.1 / nyquist
    high = 0.5 / nyquist
    b, a = butter(4, [low, high], btype="band")
    filtered = filtfilt(b, a, bvp_signal)
    freqs = np.fft.rfftfreq(len(filtered), d=1 / fs)
    spectrum = np.abs(np.fft.rfft(filtered))
    band = (freqs >= 0.1) & (freqs <= 0.5)
    if not np.any(band):
        return 0.0
    return freqs[band][np.argmax(spectrum[band])] * 60


def time_to_str(t, mode="min"):
    if mode == "min":
        t = int(t) / 60
        hr = t // 60
        minute = t % 60
        return "%2d hr %02d min" % (hr, minute)
    if mode == "sec":
        t = int(t)
        minute = t // 60
        sec = t % 60
        return "%2d min %02d sec" % (minute, sec)
    raise NotImplementedError


class Logger:
    def __init__(self):
        self.terminal = sys.stdout
        self.file = None

    def open(self, file, mode=None):
        self.file = open(file, mode or "w")

    def write(self, message, is_terminal=1, is_file=1):
        if "\r" in message:
            is_file = 0
        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file == 1 and self.file is not None:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        pass


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train or adapt GAP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-g", "--GPU", dest="GPU", type=str, default="0")
    parser.add_argument("-p", "--pp", dest="num_workers", type=int, default=2)
    parser.add_argument("-e", "--epochs", dest="epochs", type=int, default=20000)
    parser.add_argument("-b", "--batch-size", dest="batchsize", type=int, default=100)
    parser.add_argument("-l", "--learning-rate", dest="lr", type=float, default=1e-5)
    parser.add_argument("-pt", "--pretrain", dest="pt", type=str, default="resnet18")
    parser.add_argument("-rT", "--reTrain", dest="reTrain", type=int, default=0)
    parser.add_argument("-rD", "--reData", dest="reData", type=int, default=0)
    parser.add_argument("-mi", "--max_iter", dest="max_iter", type=int, default=20000)
    parser.add_argument("-s", "--seed", dest="seed", type=int, default=0)
    parser.add_argument("-tr", "--temporal_aug_rate", dest="temporal_aug_rate", type=float, default=0.1)
    parser.add_argument("-sr", "--spatial_aug_rate", dest="spatial_aug_rate", type=float, default=0.5)
    parser.add_argument("-f", "--form", dest="form", type=str, default="Resize")
    parser.add_argument("-n", "--frames_num", dest="frames_num", type=int, default=256)
    parser.add_argument("-t", "--tgt", dest="tgt", type=str, default="VIPL")
    parser.add_argument("--data-root", dest="data_root", type=str, default="./data/STMap")
    parser.add_argument("--save-dir", dest="save_dir", type=str, default="./Result_Model")
    parser.add_argument("--checkpoint", dest="checkpoint", type=str, default="")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny random-data check")
    parser.add_argument("--p1", type=float, default=0.0001)
    parser.add_argument("--p2", type=float, default=0.001)
    parser.add_argument("--p3", type=float, default=1.0)
    parser.add_argument("--p4", type=float, default=0.01)
    parser.add_argument("--p5", type=float, default=0.01)
    parser.add_argument("--p6", type=float, default=0.01)
    parser.add_argument("--p7", type=float, default=0.01)
    parser.add_argument("--pi", type=float, default=0.1)
    return parser.parse_args(argv)


def MyEval(pred, target):
    pred = np.array(pred).reshape(-1)
    target = np.array(target).reshape(-1)
    diff = pred - target
    me = np.mean(diff)
    std = np.std(diff)
    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(np.power(diff, 2)))
    mer = np.mean(np.abs(diff) / (np.abs(target) + 1e-6))
    p = np.sum((pred - np.mean(pred)) * (target - np.mean(target))) / (
        1e-6 + np.linalg.norm(pred - np.mean(pred), ord=2) * np.linalg.norm(target - np.mean(target), ord=2)
    )
    return me, std, mae, rmse, mer, p
