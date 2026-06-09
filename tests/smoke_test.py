import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch

import TTA_methods
import train
import utils
from Model import My_model


def main():
    args = utils.get_args(["--smoke", "--batch-size", "2", "--frames_num", "256"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = My_model(people_num=3, pretrained_path="").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    x = torch.randn(2, 3, 64, 256, device=device)
    xa = x + 0.01 * torch.randn_like(x)
    labels = {
        "bvp": torch.randn(2, 256, device=device),
        "hr": torch.tensor([72.0, 88.0], device=device),
        "spo2": torch.tensor([98.0, 97.0], device=device),
        "rr": torch.tensor([16.0, 19.0], device=device),
    }
    masks = {key: torch.ones(2, dtype=torch.bool, device=device) for key in labels}
    person_id = torch.tensor([0, 1], dtype=torch.long, device=device)
    loss, metrics = train.gap_mssdg_loss(model, x, xa, labels, masks, person_id, args)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("MSSDG smoke:", {k: float(v.cpu()) for k, v in metrics.items()})

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    adapter = TTA_methods.TTA_Prior(model, optimizer)
    outputs, loss_items, _ = adapter(x[:1], xa[:1], offset_label=torch.tensor([2], device=device))
    print("TTPA smoke:", [tuple(o.shape) for o in outputs], {k: float(v.cpu()) for k, v in loss_items.items()})


if __name__ == "__main__":
    main()
