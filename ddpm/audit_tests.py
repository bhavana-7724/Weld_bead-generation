import unittest
import torch
import numpy as np
import os
import pandas as pd
from ddpm.config import Config
from ddpm.dataset import WeldDataset, get_interpolation_dataset
from ddpm.model import ConditionalUNet
from ddpm.diffusion import DiffusionScheduler
from ddpm.evaluate import evaluate_metrics

class AuditTests(unittest.TestCase):

    def test_split_disjointness(self):
        train = pd.read_csv(Config.TRAIN_CSV)
        val = pd.read_csv(Config.VAL_CSV)
        test = pd.read_csv(Config.TEST_CSV)
        
        train_exps = set(train['experiment_id'])
        val_exps = set(val['experiment_id'])
        test_exps = set(test['experiment_id'])
        
        self.assertTrue(train_exps.isdisjoint(val_exps))
        self.assertTrue(train_exps.isdisjoint(test_exps))
        self.assertTrue(val_exps.isdisjoint(test_exps))
        
        interp_ids = {5, 14, 22, 32}
        self.assertTrue(interp_ids.isdisjoint(train_exps))
        self.assertTrue(interp_ids.isdisjoint(val_exps))
        self.assertTrue(interp_ids.isdisjoint(test_exps))

    def test_ab_integrity(self):
        for csv_path in [Config.TRAIN_CSV, Config.VAL_CSV, Config.TEST_CSV]:
            df = pd.read_csv(csv_path)
            counts = df.groupby('experiment_id').size()
            self.assertTrue(all(counts == 2), f"A/B integrity failed in {csv_path}")

    def test_dataset_loading_normalization(self):
        ds = WeldDataset(Config.TRAIN_CSV, Config.PROCESSED_IMG_DIR, is_train=False)
        img_tensor, cond_tensor, exp_id, img_id = ds[0]
        
        self.assertEqual(img_tensor.shape, (1, 256, 256))
        self.assertTrue((img_tensor >= -1.0).all() and (img_tensor <= 1.0).all())
        
        # 4 continuous, 4 categorical = 8
        self.assertEqual(cond_tensor.shape, (8,))
        
    def test_diffusion_scheduler(self):
        sched = DiffusionScheduler(timesteps=10, device='cpu')
        self.assertEqual(len(sched.betas), 10)
        self.assertEqual(len(sched.alphas_cumprod), 10)
        self.assertTrue(torch.all(sched.betas >= 0) and torch.all(sched.betas <= 1))

    def test_model_forward_pass_and_loss(self):
        model = ConditionalUNet(
            in_channels=1, base_channels=8, channel_mults=(1, 2),
            cond_dim=8, time_emb_dim=16
        )
        x = torch.randn(2, 1, 32, 32)
        t = torch.randint(0, 10, (2,))
        c = torch.randn(2, 8)
        
        out = model(x, t, c)
        self.assertEqual(out.shape, x.shape)
        
        loss = torch.nn.functional.mse_loss(out, torch.randn_like(out))
        self.assertTrue(torch.isfinite(loss))

    def test_backprop_and_optimizer(self):
        model = ConditionalUNet(in_channels=1, base_channels=8, channel_mults=(1, 2), cond_dim=8, time_emb_dim=16)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        x = torch.randn(2, 1, 32, 32)
        t = torch.tensor([1, 2])
        c = torch.randn(2, 8)
        out = model(x, t, c)
        
        loss = out.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        # Check if gradients exist
        has_grad = False
        for param in model.parameters():
            if param.grad is not None:
                has_grad = True
                break
        self.assertTrue(has_grad)

    def test_checkpoint_roundtrip(self):
        model1 = ConditionalUNet(in_channels=1, base_channels=8, channel_mults=(1, 2), cond_dim=8, time_emb_dim=16)
        path = os.path.join(Config.MODEL_DIR, "test_chk.pt")
        torch.save({'model_state_dict': model1.state_dict()}, path)
        
        model2 = ConditionalUNet(in_channels=1, base_channels=8, channel_mults=(1, 2), cond_dim=8, time_emb_dim=16)
        model2.load_state_dict(torch.load(path)['model_state_dict'])
        
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.allclose(p1, p2))
            
        os.remove(path)

    def test_metric_behavior(self):
        img1 = np.ones((256, 256), dtype=np.uint8) * 128
        img2 = np.ones((256, 256), dtype=np.uint8) * 128
        
        mae, rmse, s, p = evaluate_metrics(img1, img2)
        self.assertEqual(mae, 0.0)
        self.assertEqual(rmse, 0.0)
        self.assertAlmostEqual(s, 1.0)
        self.assertEqual(p, float('inf'))
        
        img3 = np.ones((256, 256), dtype=np.uint8) * 100
        mae2, rmse2, s2, p2 = evaluate_metrics(img1, img3)
        self.assertEqual(mae2, 28.0)
        self.assertEqual(rmse2, 28.0)
        self.assertTrue(p2 < float('inf'))

if __name__ == '__main__':
    unittest.main(verbosity=2)
