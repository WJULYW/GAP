import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

import MyDataset
import MyLoss
import Model
import utils


TARGET_DOMAIN = {
    "VIPL": ["PURE", "BUAA", "UBFC", "V4V", "HCW"],
    "V4V": ["VIPL", "PURE", "BUAA", "UBFC", "HCW"],
    "PURE": ["VIPL", "BUAA", "UBFC", "V4V", "HCW"],
    "BUAA": ["VIPL", "PURE", "UBFC", "V4V", "HCW"],
    "UBFC": ["VIPL", "PURE", "BUAA", "V4V", "HCW"],
    "HCW": ["VIPL", "PURE", "BUAA", "UBFC", "V4V"],
    "HMPC-Dv1": ["VIPL", "PURE", "BUAA", "UBFC", "V4V", "HCW"],
}

FILE_NAME = {
    "VIPL": ["VIPL", "VIPL", "STMap_RGB_Align_CSI.png"],
    "V4V": ["V4V", "V4V", "STMap_RGB.png"],
    "PURE": ["PURE", "PURE", "STMap.png"],
    "BUAA": ["BUAA", "BUAA", "STMap_RGB.png"],
    "UBFC": ["UBFC", "UBFC", "STMap.png"],
    "HCW": ["HCW", "HCW", "STMap_RGB.png"],
    "HMPC-Dv1": ["HMPC-Dv1", "HMPC-Dv1", "STMap_RGB.png"],
    "VV100": ["VV100", "VV100", "STMap_RGB.png"],
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_device(args):
    if torch.cuda.is_available():
        return torch.device(f"cuda:{args.GPU}")
    return torch.device("cpu")


def dataset_paths(data_root, dataset_name):
    file_root, index_name, stmap = FILE_NAME[dataset_name]
    return (
        os.path.join(data_root, file_root),
        os.path.join(data_root, "STMap_Index", index_name),
        stmap,
    )


def collect_people(data_root, source_names):
    people = []
    for name in source_names:
        root, _, _ = dataset_paths(data_root, name)
        if os.path.isdir(root):
            people.extend(sorted(os.listdir(root)))
    return sorted(set(people))


def maybe_build_index(data_root, dataset_name, args):
    file_root, index_root, stmap = dataset_paths(data_root, dataset_name)
    if os.path.isdir(index_root) and len(os.listdir(index_root)) > 0:
        return
    if args.reData != 1:
        raise FileNotFoundError(
            f"Missing STMap index directory: {index_root}. "
            "Run with --reData 1 to build indexes from raw STMap folders."
        )
    if not os.path.isdir(file_root):
        raise FileNotFoundError(f"Missing dataset directory: {file_root}")
    MyDataset.getIndex(file_root, os.listdir(file_root), index_root, stmap, Step=10, frames_num=args.frames_num)


def build_source_loaders(args, source_names, people_list):
    loaders = []
    per_source_batch = max(1, args.batchsize // max(len(source_names), 1))
    for domain_id, name in enumerate(source_names):
        maybe_build_index(args.data_root, name, args)
        _, index_root, stmap = dataset_paths(args.data_root, name)
        dataset = MyDataset.Data_DG(
            root_dir=index_root,
            dataName=name,
            STMap=stmap,
            frames_num=args.frames_num,
            args=args,
            domain_label=domain_id,
            output_people=True,
            peoplelist=people_list,
        )
        loaders.append(
            (
                name,
                DataLoader(
                    dataset,
                    batch_size=per_source_batch,
                    shuffle=True,
                    num_workers=args.num_workers,
                    drop_last=True,
                ),
            )
        )
    return loaders


def labels_and_masks_from_batches(batches, names, device):
    xs, xas, person_ids = [], [], []
    labels = {"bvp": [], "hr": [], "spo2": [], "rr": []}
    masks = {"bvp": [], "hr": [], "spo2": [], "rr": []}

    for batch, name in zip(batches, names):
        x, bvp, hr, spo2, rr, xa, *_rest, person_id = batch
        batch_size = x.shape[0]
        availability = MyDataset.DATASET_LABELS.get(name, {"bvp": True, "hr": True, "spo2": True, "rr": True})

        xs.append(x.float())
        xas.append(xa.float())
        person_ids.append(person_id.long())
        labels["bvp"].append(bvp.float())
        labels["hr"].append(hr.float())
        labels["spo2"].append(spo2.float())
        labels["rr"].append(rr.float())
        for key in masks:
            masks[key].append(torch.full((batch_size,), availability[key], dtype=torch.bool))

    x = torch.cat(xs, dim=0).to(device)
    xa = torch.cat(xas, dim=0).to(device)
    person_id = torch.cat(person_ids, dim=0).to(device)
    labels = {key: torch.cat(value, dim=0).to(device) for key, value in labels.items()}
    masks = {key: torch.cat(value, dim=0).to(device) for key, value in masks.items()}
    return x, xa, labels, masks, person_id


def gap_mssdg_loss(model, x, xa, labels, masks, person_id, args):
    out_o = model(x, mode="mssdg", return_details=True)
    out_a = model(xa, mode="mssdg", return_details=True)

    lssa, lsda = MyLoss.new_style_alignment_loss(out_o["block_feats"], out_a["block_feats"])
    lmt = MyLoss.supervised_multitask_loss(out_o["preds"], labels, masks)
    lp = MyLoss.person_classification_loss(out_o["person_logits"], person_id)
    lfc = MyLoss.frequency_consistency_loss(out_o["preds"], out_a["preds"])
    ltc = MyLoss.temporal_consistency_loss(out_o["preds"]["bvp"], out_a["preds"]["bvp"])
    lpe = MyLoss.person_elimination_loss(out_o["task_reprs"], out_o["person_repr"], pi=args.pi)

    loss = args.p1 * lssa + args.p2 * lsda + args.p3 * lp + args.p4 * lfc + lmt + args.p5 * lpe + args.p6 * ltc
    metrics = {
        "loss": loss.detach(),
        "LMT": lmt.detach(),
        "LSSA": lssa.detach(),
        "LSDA": lsda.detach(),
        "LP": lp.detach(),
        "LFC": lfc.detach(),
        "LPE": lpe.detach(),
        "LTC": ltc.detach(),
    }
    return loss, metrics


def run_smoke(args, device):
    model = Model.My_model(people_num=3, pretrained_path="").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    x = torch.randn(2, 3, 64, args.frames_num, device=device)
    xa = x + 0.01 * torch.randn_like(x)
    labels = {
        "bvp": torch.randn(2, args.frames_num, device=device),
        "hr": torch.tensor([70.0, 82.0], device=device),
        "spo2": torch.tensor([98.0, 97.0], device=device),
        "rr": torch.tensor([16.0, 18.0], device=device),
    }
    masks = {key: torch.ones(2, dtype=torch.bool, device=device) for key in labels}
    person_id = torch.tensor([0, 1], dtype=torch.long, device=device)
    loss, metrics = gap_mssdg_loss(model, x, xa, labels, masks, person_id, args)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Smoke MSSDG step ok:", {key: float(value.cpu()) for key, value in metrics.items()})


def main():
    args = utils.get_args()
    set_seed(args.seed)
    device = build_device(args)

    if args.smoke:
        run_smoke(args, device)
        return

    source_names = TARGET_DOMAIN[args.tgt]
    people_list = collect_people(args.data_root, source_names)
    people_num = len(people_list) + 1
    loaders = build_source_loaders(args, source_names, people_list)
    iterators = [(name, loader, iter(loader)) for name, loader in loaders]

    model = Model.My_model(people_num=people_num).to(device)
    model.calculate_training_parameter_ratio()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    model.train()
    for iter_num in range(args.max_iter):
        names, batches = [], []
        refreshed_iterators = []
        for name, loader, iterator in iterators:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            names.append(name)
            batches.append(batch)
            refreshed_iterators.append((name, loader, iterator))
        iterators = refreshed_iterators
        x, xa, labels, masks, person_id = labels_and_masks_from_batches(batches, names, device)
        loss, metrics = gap_mssdg_loss(model, x, xa, labels, masks, person_id, args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iter_num % 50 == 0:
            msg = " ".join([f"{key}:{float(value.cpu()):.4f}" for key, value in metrics.items()])
            print(f"[{iter_num}/{args.max_iter}] {msg}")

    save_path = os.path.join(args.save_dir, f"GAP_G_{args.tgt}.pth")
    torch.save({"model": model.state_dict(), "people_num": people_num, "people_list": people_list}, save_path)
    print(f"Saved checkpoint to {save_path}")


if __name__ == "__main__":
    main()
