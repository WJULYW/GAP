import torch
import torch.nn as nn

import MyLoss


class TTA_Prior(nn.Module):
    """Test-time personalized adaptation wrapper for GAP-P."""

    def __init__(
        self,
        model,
        optimizer,
        steps=1,
        device=None,
        p1=0.0001,
        p2=0.001,
        p3=1.0,
        p4=0.01,
        p7=0.01,
    ):
        super().__init__()
        if steps <= 0:
            raise ValueError("TTA_Prior requires at least one adaptation step")
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.device = device
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4
        self.p7 = p7

    def set_prior_loss(self, *args, **kwargs):
        """Kept for compatibility with old scripts."""
        return None

    def forward(self, x, x_aug, iter_num=0, offset_label=None):
        loss_items = None
        for _ in range(self.steps):
            outputs, loss_items = forward_and_adapt(
                x,
                x_aug,
                self.model,
                self.optimizer,
                offset_label=offset_label,
                p1=self.p1,
                p2=self.p2,
                p3=self.p3,
                p4=self.p4,
                p7=self.p7,
            )
        return outputs, loss_items, None


@torch.enable_grad()
def forward_and_adapt(x, x_aug, model, optimizer, offset_label=None, p1=0.0001, p2=0.001, p3=1.0, p4=0.01, p7=0.01):
    model.train()
    optimizer.zero_grad()

    out_o = model(x, mode="ttpa", return_details=True)
    out_a = model(x_aug, mode="ttpa", return_details=True)

    lssa, lsda = MyLoss.new_style_alignment_loss(out_o["block_feats"], out_a["block_feats"])
    lfc = MyLoss.frequency_consistency_loss(out_o["preds"], out_a["preds"])
    ltic = MyLoss.temporal_inconsistency_loss(out_o["preds"]["bvp"], out_a["preds"]["bvp"])

    lp = x.new_tensor(0.0)
    if offset_label is not None:
        lp = MyLoss.person_classification_loss(out_o["person_logits"], offset_label.to(x.device))

    loss = p1 * lssa + p2 * lsda + p3 * lp + p4 * lfc + p7 * ltic
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        outputs = model(x, mode="ttpa", return_details=False)

    loss_items = {
        "loss": loss.detach(),
        "LSSA": lssa.detach(),
        "LSDA": lsda.detach(),
        "LP": lp.detach(),
        "LFC": lfc.detach(),
        "LTIC": ltic.detach(),
    }
    return outputs, loss_items
