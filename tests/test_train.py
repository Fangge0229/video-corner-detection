import torch

from model import VideoCornerModel
from train import average_stats, train_one_step


def test_average_stats_accepts_list_of_dicts():
    stats = average_stats(
        [
            {"loss": 1.0, "loss_corner": 0.5, "loss_conf": 1.0},
            {"loss": 3.0, "loss_corner": 1.5, "loss_conf": 2.0},
        ]
    )

    assert stats == {"loss": 2.0, "loss_corner": 1.0, "loss_conf": 1.5}


def test_train_one_step_runs_minimal_mainline_batch():
    model = VideoCornerModel(feat_dim=32, nhead=4, num_layers=1, max_len=4)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    batch = {
        "roi_images": torch.randn(1, 4, 3, 64, 64),
        "target_corners": torch.zeros(1, 8, 2),
        "target_vis": torch.ones(1, 8),
    }

    stats = train_one_step(model, batch, optimizer, torch.device("cpu"))

    assert set(stats) == {"loss", "loss_corner", "loss_conf"}
    assert all(isinstance(value, float) for value in stats.values())
