import torch

from model import VideoCornerModel


def test_video_corner_model_forward_uses_mainline_keys_and_shapes():
    model = VideoCornerModel(feat_dim=32, nhead=4, num_layers=1, max_len=4)
    model.eval()

    with torch.no_grad():
        out = model(torch.randn(2, 4, 3, 64, 64))

    assert set(out) == {"corners_pred", "conf_logits_pred"}
    assert out["corners_pred"].shape == (2, 8, 2)
    assert out["conf_logits_pred"].shape == (2, 8)
