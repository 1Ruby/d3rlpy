import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteQFunction(nn.Module):
    def __init__(self, head, action_size):
        super().__init__()
        self.action_size = action_size
        self.head = head
        self.fc = nn.Linear(head.feature_size, action_size)

    def forward(self, x):
        h = self.head(x)
        return self.fc(h)

    def compute_td(self, obs_t, act_t, rew_tp1, q_tp1, gamma=0.99):
        one_hot = F.one_hot(act_t.view(-1), num_classes=self.action_size)
        q_t = (self.forward(obs_t) * one_hot).sum(dim=1, keepdims=True)
        y = rew_tp1 + gamma * q_tp1
        return F.smooth_l1_loss(q_t, y)


class EnsembleDiscreteQFunction(nn.Module):
    def __init__(self, heads, action_size):
        super().__init__()
        self.action_size = action_size
        self.q_functions = nn.ModuleList()
        for head in heads:
            self.q_functions.append(DiscreteQFunction(head, action_size))

    def forward(self, x, reduction='min'):
        values = []
        for q_function in self.q_functions:
            values.append(q_function(x))
        values = torch.cat(values, dim=0)

        if reduction == 'min':
            return values.min(dim=0)
        elif reduction == 'max':
            return values.max(dim=0)
        elif reduction == 'mean':
            return values.mean(dim=0)
        elif reduction == 'none':
            return values
        else:
            raise ValueError

    def compute_td(self, obs_t, act_t, rew_tp1, q_tp1, gamma=0.99):
        tds = []
        for q_function in self.q_functions:
            td = q_function.compute_td(obs_t, act_t, rew_tp1, q_tp1, gamma)
            tds.append(td)
        tds = torch.cat(tds, dim=0)
        return tds.sum()


class ContinuousQFunction(nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head
        self.fc = nn.Linear(head.feature_size, 1)

    def forward(self, x, action):
        h = self.head(x, action)
        return self.fc(h)

    def compute_td(self, obs_t, act_t, rew_tp1, q_tp1, gamma=0.99):
        q_t = self.forward(obs_t, act_t)
        y = rew_tp1 + gamma * q_tp1
        return F.mse_loss(q_t, y)


class EnsembleContinuousQFunction(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.q_functions = nn.ModuleList()
        for head in heads:
            self.q_functions.append(ContinuousQFunction(head))

    def forward(self, x, action, reduction='min'):
        values = []
        for q_function in self.q_functions:
            values.append(q_function(x, action))
        values = torch.cat(values, dim=0)

        if reduction == 'min':
            return values.min(dim=0)
        elif reduction == 'max':
            return values.max(dim=0)
        elif reduction == 'mean':
            return values.mean(dim=0)
        else:
            raise ValueError

    def compute_td(self, obs_t, act_t, rew_tp1, q_tp1, gamma=0.99):
        tds = []
        for q_function in self.q_functions:
            td = q_function.compute_td(obs_t, act_t, rew_tp1, q_tp1, gamma)
            tds.append(td)
        tds = torch.cat(tds, dim=0)
        return tds.sum()
