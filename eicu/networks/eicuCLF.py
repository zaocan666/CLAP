import torch
import torch.nn as nn
import torch.nn.functional as F

class eicuCLF_lstm(nn.Module):
    def __init__(self, num_classes):
        super(eicuCLF_lstm, self).__init__()
        hidden_dim = 16
        dropout = 0
        self.bilstm = nn.LSTM(1, hidden_dim, num_layers=2, dropout=dropout, bidirectional=True, bias=False)
        self.linear2 = nn.Linear( 2*hidden_dim, num_classes)

    def forward(self, x):
        x = torch.unsqueeze(x, -1) # [B, 48, 1]
        x = torch.transpose(x, 0,1)
        self.bilstm.flatten_parameters()
        bilstm_out, _ = self.bilstm(x) # [L, N, 2*hidden_dim]
        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2) # [N, 2*hidden_dim, L]
        bilstm_out = torch.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2) # [N, 2*hidden_dim]
        logit = self.linear2(bilstm_out)
        return logit
