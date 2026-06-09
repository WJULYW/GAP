import os
import random
from collections import defaultdict

import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import DataLoader

import Model
import MyDataset
import TTA_methods
import utils
from train import FILE_NAME, build_device, dataset_paths, maybe_build_index, set_seed


def _mat_string(value):
    arr = np.asarray(value).reshape(-1)
    return str(arr[0]) if len(arr) else ""


def load_gap_checkpoint(path, device):
    if not path:
        raise ValueError("--checkpoint is required for TTPA")
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"], checkpoint.get("people_num", 1), checkpoint.get("people_list", [])
    return checkpoint, 1, []


def group_indices_by_subject(index_root):
    groups = defaultdict(list)
    for name in sorted(os.listdir(index_root)):
        if not name.endswith(".mat"):
            continue
        item = scio.loadmat(os.path.join(index_root, name))
        path = _mat_string(item["Path"])
        subject = os.path.basename(path)
        groups[subject].append(name)
    return groups


def run_smoke(args, device):
    model = Model.My_model(people_num=3, pretrained_path="").to(device)
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    for _subject in range(2):
        model.load_state_dict(base_state)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        adapter = TTA_methods.TTA_Prior(
            model,
            optimizer,
            p1=args.p1,
            p2=args.p2,
            p3=args.p3,
            p4=args.p4,
            p7=args.p7,
        )
        x = torch.randn(1, 3, 64, args.frames_num, device=device)
        xa = x + 0.01 * torch.randn_like(x)
        offset_label = torch.tensor([2], dtype=torch.long, device=device)
        outputs, loss_items, _ = adapter(x, xa, offset_label=offset_label)
        print("Smoke TTPA step ok:", [tuple(o.shape) for o in outputs], {k: float(v.cpu()) for k, v in loss_items.items()})


def main():
    args = utils.get_args()
    set_seed(args.seed)
    random.seed(args.seed)
    device = build_device(args)

    if args.smoke:
        run_smoke(args, device)
        return

    state_dict, people_num, people_list = load_gap_checkpoint(args.checkpoint, device)
    model = Model.My_model(people_num=people_num).to(device)
    model.load_state_dict(state_dict, strict=False)
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    maybe_build_index(args.data_root, args.tgt, args)
    _, index_root, stmap = dataset_paths(args.data_root, args.tgt)
    subject_groups = group_indices_by_subject(index_root)
    os.makedirs(args.save_dir, exist_ok=True)

    unseen_person_id = max(people_num - 1, 0)
    for subject, datalist in subject_groups.items():
        model.load_state_dict(base_state)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        adapter = TTA_methods.TTA_Prior(
            model,
            optimizer,
            p1=args.p1,
            p2=args.p2,
            p3=args.p3,
            p4=args.p4,
            p7=args.p7,
        )
        dataset = MyDataset.Data_DG(
            root_dir=index_root,
            dataName=args.tgt,
            STMap=stmap,
            frames_num=args.frames_num,
            args=args,
            domain_label=unseen_person_id,
            datalist=datalist,
            peoplelist=people_list,
            output_people=False,
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
        predictions = []
        for batch in loader:
            x, _, _, _, _, xa, *_ = batch
            x = x.float().to(device)
            xa = xa.float().to(device)
            offset_label = torch.full((x.shape[0],), unseen_person_id, dtype=torch.long, device=device)
            outputs, _, _ = adapter(x, xa, offset_label=offset_label)
            bvp, hr, spo2, rr = outputs
            predictions.append(
                {
                    "hr": hr.detach().cpu().numpy(),
                    "spo2": spo2.detach().cpu().numpy(),
                    "rr": rr.detach().cpu().numpy(),
                    "bvp": bvp.detach().cpu().numpy(),
                }
            )
        torch.save(predictions, os.path.join(args.save_dir, f"{subject}_ttpa_predictions.pt"))
        print(f"Finished TTPA subject {subject} with {len(predictions)} samples")


if __name__ == "__main__":
    main()
