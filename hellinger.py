import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # output 1 score

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class HellingerMIEstimator:
    def __init__(self, input_dim, hidden=256, lr=5e-4, weight_decay=0.0, hidden_layer=(256, 256), device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = MLP(input_dim , hidden).to(self.device)  # x_dim + y_dim input
        self.opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.eps = 1e-6

    def _to_pos(self, s, score_clip):
        
        if score_clip is not None:
            s = torch.clamp(s, -score_clip, score_clip)
        D = F.softplus(s) + self.eps  # D > 0
        invD = 1.0 / D  # D^{-1}
        return D, invD

    @torch.no_grad()
    def estimate(self, x, y):
        self.model.eval()
        x = x.to(self.device); y = y.to(self.device)
        xy = torch.cat([x, y], dim=1)
        y_perm = y[torch.randperm(y.shape[0])]
        xy_perm = torch.cat([x, y_perm], dim=1)
        s_joint = self.model(xy)
        s_prod = self.model(xy_perm)
        J = 2.0 - torch.mean(torch.exp(s_joint)) - torch.mean(torch.exp(-s_prod))
        return J.item()

    def compute_J(self, x, y, k_shuffle, detach=False, score_clip=None):
        if detach:
            x, y = x.detach(), y.detach()
        x = x.to(self.device)
        y = y.to(self.device)

        xy = torch.cat([x, y], dim=1)
        s_joint = self.model(xy)
        D_joint, _ = self._to_pos(s_joint, score_clip)  # E_P[D]
        EP_D = D_joint.mean()

        eq_list = []
        for _ in range(k_shuffle):
            y_perm = y[torch.randperm(y.shape[0], device=y.device)]
            s_prod = self.model(torch.cat([x, y_perm], dim=1))
            _, invD = self._to_pos(s_prod, score_clip)  # E_Q[D^{-1}]
            eq_list.append(invD.mean())

        EQ_inv = torch.stack(eq_list).mean()
        return 2.0 - EP_D - EQ_inv

    def train_step(self, x, y, k_shuffle, score_clip):
        self.model.train()
        x = x.to(self.device); y = y.to(self.device)

        J = self.compute_J(x, y, k_shuffle=k_shuffle, detach=False, score_clip=score_clip)
        loss = -J  # maximize J
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
        self.opt.step()
        return J.item()