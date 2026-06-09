import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-6


def _zero_like_reference(reference):
    return reference.new_tensor(0.0)


def _squeeze_scalar(x):
    if x is None:
        return None
    return x.reshape(x.shape[0], -1).mean(dim=1)


def _squeeze_wave(x):
    if x.dim() == 3:
        return x.squeeze(1)
    return x


class P_loss3(nn.Module):
    """Negative Pearson-style BVP loss used in the original training code."""

    def forward(self, pred, target):
        pred = _squeeze_wave(pred)
        target = _squeeze_wave(target)
        pred = pred - pred.mean(dim=-1, keepdim=True)
        target = target - target.mean(dim=-1, keepdim=True)
        corr = (pred * target).sum(dim=-1) / (
            torch.sqrt((pred ** 2).sum(dim=-1) + EPS)
            * torch.sqrt((target ** 2).sum(dim=-1) + EPS)
        )
        return 1.0 - corr.mean()


def negative_pearson_loss(pred, target):
    return P_loss3()(pred, target)


def flatten_block_feature(feat, normalize=True):
    """Convert an encoder block feature to the matrix used by LSSA/LSDA."""
    if feat.dim() != 4:
        raise ValueError(f"Expected [B,C,H,W] feature, got {tuple(feat.shape)}")
    # Average over the ROI-like spatial axis and keep temporal/channel structure.
    z = feat.mean(dim=2).flatten(1)
    if normalize:
        z = F.layer_norm(z, z.shape[1:])
    return z


def semantic_structure_alignment_loss(features_o, features_a):
    """LSSA: exchange singular values between original and augmented features."""
    loss = _zero_like_reference(features_o[0])
    for feat_o, feat_a in zip(features_o, features_a):
        z_o = flatten_block_feature(feat_o)
        z_a = flatten_block_feature(feat_a)
        u_o, s_o, vh_o = torch.linalg.svd(z_o, full_matrices=False)
        u_a, s_a, vh_a = torch.linalg.svd(z_a, full_matrices=False)
        r_oa = u_o @ torch.diag_embed(s_a) @ vh_o
        r_ao = u_a @ torch.diag_embed(s_o) @ vh_a
        loss = loss + torch.linalg.norm(r_oa - r_ao, ord="fro") / z_o.shape[0]
    return loss


def semantic_distribution_alignment_loss(features_o, features_a):
    """LSDA: align batch-wise Gram distributions of original and augmented features."""
    loss = _zero_like_reference(features_o[0])
    for feat_o, feat_a in zip(features_o, features_a):
        z_o = F.normalize(flatten_block_feature(feat_o), dim=1)
        z_a = F.normalize(flatten_block_feature(feat_a), dim=1)
        gram_o = z_o @ z_o.t()
        gram_a = z_a @ z_a.t()
        loss = loss + torch.linalg.norm(gram_o - gram_a, ord="fro") / z_o.shape[0]
    return loss


def new_style_alignment_loss(features_o, features_a, alpha=None):
    """Backward-compatible wrapper returning ``(LSSA, LSDA)``."""
    return (
        semantic_structure_alignment_loss(features_o, features_a),
        semantic_distribution_alignment_loss(features_o, features_a),
    )


def _bvp_psd_distribution(bvp, fps=30.0, low_hz=40 / 60, high_hz=180 / 60):
    bvp = _squeeze_wave(bvp)
    bvp = bvp - bvp.mean(dim=-1, keepdim=True)
    window = torch.hann_window(bvp.shape[-1], device=bvp.device, dtype=bvp.dtype)
    spectrum = torch.fft.rfft(bvp * window, dim=-1).abs().pow(2)
    freqs = torch.fft.rfftfreq(bvp.shape[-1], d=1.0 / fps).to(bvp.device)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    spectrum = spectrum[:, band]
    return spectrum / (spectrum.sum(dim=-1, keepdim=True) + EPS)


def frequency_consistency_loss(pred_o, pred_a):
    """LFC over HR, BVP, SpO2 and RR predictions."""
    ref = next(v for v in pred_o.values() if v is not None)
    loss = _zero_like_reference(ref)
    for key in ("hr", "spo2", "rr"):
        if pred_o.get(key) is not None and pred_a.get(key) is not None:
            loss = loss + F.l1_loss(_squeeze_scalar(pred_o[key]), _squeeze_scalar(pred_a[key]))
    if pred_o.get("bvp") is not None and pred_a.get("bvp") is not None:
        q_o = _bvp_psd_distribution(pred_o["bvp"])
        q_a = _bvp_psd_distribution(pred_a["bvp"])
        loss = loss + (q_a * ((q_a + EPS).log() - (q_o + EPS).log())).sum(dim=-1).mean()
    return loss


def _self_similarity_matrix(bvp, window_size=10):
    bvp = _squeeze_wave(bvp)
    if bvp.shape[-1] < window_size:
        raise ValueError("BVP length must be >= window_size")
    windows = bvp.unfold(dimension=-1, size=window_size, step=1)
    windows = F.normalize(windows, p=2, dim=-1)
    return windows @ windows.transpose(1, 2)


def temporal_inconsistency_loss(bvp_o, bvp_a, window_size=10):
    """LTIC from the paper: cosine similarity of BVP self-similarity matrices."""
    sim_o = _self_similarity_matrix(bvp_o, window_size).flatten(1)
    sim_a = _self_similarity_matrix(bvp_a, window_size).flatten(1)
    return F.cosine_similarity(sim_o, sim_a, dim=1).mean()


def temporal_consistency_loss(bvp_o, bvp_a, window_size=10):
    return 1.0 - temporal_inconsistency_loss(bvp_o, bvp_a, window_size)


def cosine_similarity_loss(matrix1, matrix2):
    matrix1_flat = matrix1.reshape(matrix1.size(0), -1)
    matrix2_flat = matrix2.reshape(matrix2.size(0), -1)
    return F.cosine_similarity(matrix1_flat, matrix2_flat, dim=1).mean()


def person_classification_loss(logits, person_id):
    return F.cross_entropy(logits, person_id.long())


def person_elimination_loss(task_reprs, person_repr, pi=0.1):
    """LPE from the paper.

    The paper writes ``||Zp Zi^T - I||`` with a lower-bound constraint. This is
    kept as-is for method fidelity, although many orthogonality penalties use a
    zero matrix instead of the identity matrix.
    """
    identity = torch.eye(person_repr.shape[0], device=person_repr.device, dtype=person_repr.dtype)
    loss = person_repr.new_tensor(0.0)
    for repr_i in task_reprs.values():
        z_p = F.normalize(person_repr, dim=1)
        z_i = F.normalize(repr_i, dim=1)
        loss = loss + torch.linalg.norm(z_p @ z_i.t() - identity, ord="fro") / z_p.shape[0]
    loss = loss / max(len(task_reprs), 1)
    return torch.clamp_min(loss, pi)


def supervised_multitask_loss(preds, labels, masks=None):
    """LMT with partial-label masks.

    ``labels`` may contain ``hr``, ``bvp``, ``spo2`` and ``rr`` entries. Missing
    labels can be omitted or set to ``None``.
    """
    ref = next(v for v in preds.values() if v is not None)
    loss = _zero_like_reference(ref)
    count = 0
    masks = masks or {}

    def selected(key):
        label = labels.get(key)
        if label is None:
            return None, None
        mask = masks.get(key)
        if mask is None:
            mask = torch.ones(label.shape[0], dtype=torch.bool, device=label.device)
        else:
            mask = mask.bool().to(label.device)
        if mask.sum() == 0:
            return None, None
        return label[mask], mask

    for key in ("hr", "spo2", "rr"):
        label, mask = selected(key)
        if label is not None:
            pred = _squeeze_scalar(preds[key][mask])
            loss = loss + F.l1_loss(pred, label.reshape(-1).float())
            count += 1

    label, mask = selected("bvp")
    if label is not None:
        loss = loss + negative_pearson_loss(preds["bvp"][mask], label.float())
        count += 1

    return loss / max(count, 1)


class ConsistencyLoss(nn.Module):
    """Backward-compatible LFC wrapper used by TTA_methods."""

    def forward(self, hr, hr_aug, rr, rr_aug, spo, spo_aug, bvp, bvp_aug, *args, **kwargs):
        preds = {"hr": hr, "rr": rr, "spo2": spo, "bvp": bvp}
        preds_aug = {"hr": hr_aug, "rr": rr_aug, "spo2": spo_aug, "bvp": bvp_aug}
        total = frequency_consistency_loss(preds, preds_aug)
        bvp_part = frequency_consistency_loss({"bvp": bvp}, {"bvp": bvp_aug})
        return total - bvp_part, bvp_part


def get_HR_from_bvp_torch(signals, fs=30, harmonics_removal=True):
    signals = _squeeze_wave(signals)
    signal_length = signals.shape[-1]
    spectrum = torch.fft.rfft(signals * torch.hann_window(signal_length, device=signals.device), dim=1).abs()
    freqs = torch.fft.rfftfreq(signal_length, d=1.0 / fs).to(signals.device)
    low = int(round(0.6 / fs * signal_length))
    high = int(round(3.0 / fs * signal_length))
    spectrum[:, :low] = 0
    spectrum[:, high:] = 0
    peak = spectrum.argmax(dim=1)
    return freqs[peak] * 60, spectrum, freqs * 60


def get_RR_from_bvp_torch(bvp_signals, fs=30):
    bvp_signals = _squeeze_wave(bvp_signals)
    signal_length = bvp_signals.shape[-1]
    spectrum = torch.fft.rfft(bvp_signals, dim=1).abs().pow(2)
    freqs = torch.fft.rfftfreq(signal_length, d=1.0 / fs).to(bvp_signals.device)
    band = (freqs >= 0.1) & (freqs <= 0.5)
    peak = spectrum[:, band].argmax(dim=1)
    return freqs[band][peak] * 60


class bvp_hr_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, pre_bvp, gt_hr):
        hrs, _, _ = get_HR_from_bvp_torch(pre_bvp)
        return self.l1(hrs, gt_hr.reshape(-1).float())


class bvp_rr_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, gt_bvp, pre_rr):
        rr = get_RR_from_bvp_torch(gt_bvp)
        return self.l1(pre_rr.reshape(-1).float(), rr)


class Asp_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

    def forward(self, spo_pred, spo):
        return self.l1(spo_pred.reshape(-1).float(), spo.reshape(-1).float())
