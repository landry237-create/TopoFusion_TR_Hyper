import torch

from training.trainer import Trainer


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)


def test_trainer_accepts_string_numeric_training_config():
    config = {
        'training': {
            'lr': '1e-4',
            'weight_decay': '1e-5',
            'lambda_sem': '1.0',
            'lambda_geom': '0.5',
            'lambda_top': '0.1',
            'lambda_rob': '0.1',
            'lambda_temp': '0.1',
        }
    }

    trainer = Trainer(DummyModel(), [], config, device='cpu')

    assert trainer.optimizer.param_groups[0]['lr'] == 1e-4
    assert trainer.optimizer.param_groups[0]['weight_decay'] == 1e-5
