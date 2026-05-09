import torch

from loss import corner_loss


def test_corner_loss_returns_mainline_keys_and_backward_works():
    pred_corners = torch.randn(2, 8, 2, requires_grad=True)
    target_corners = torch.zeros(2, 8, 2)
    pred_conf_logits = torch.randn(2, 8, requires_grad=True)
    target_vis = torch.ones(2, 8)

    losses = corner_loss(pred_corners, target_corners, pred_conf_logits, target_vis)
    losses["loss"].backward()

    assert set(losses) == {"loss", "loss_corner", "loss_conf"}
    assert pred_corners.grad is not None
    assert pred_conf_logits.grad is not None
