import torch
import torch.nn as nn
import torch.nn.functional as F

class eicu_encoder(nn.Module):
    def __init__(self, flags):
        super(eicu_encoder, self).__init__()
        hidden_dim = 16
        dropout = 0
        self.bilstm = nn.LSTM(1, hidden_dim, num_layers=2, dropout=dropout, bidirectional=True, bias=False)
        self.linear2 = nn.Sequential(
            nn.Linear( 2*hidden_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Linear( 2*hidden_dim, 2*hidden_dim),
            nn.ReLU()
        )
        
        self.class_mu = nn.Linear(2*hidden_dim, flags.class_dim)
        self.class_logvar = nn.Linear(2*hidden_dim, flags.class_dim)
        assert not flags.factorized_representation

    def forward(self, x):
        x = torch.unsqueeze(x, -1) # [B, 48, 1]
        x = torch.transpose(x, 0,1)
        self.bilstm.flatten_parameters()
        bilstm_out, _ = self.bilstm(x) # [L, N, 2*hidden_dim]
        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2) # [N, 2*hidden_dim, L]
        bilstm_out = torch.tanh(bilstm_out)
        bilstm_out = F.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2) # [N, 2*hidden_dim]
        h = self.linear2(bilstm_out)
        mu = self.class_mu(h)
        logvar = self.class_logvar(h)
        return None, None, mu, logvar

class eicu_decoder(nn.Module):
    def __init__(self, flags):
        super(eicu_decoder, self).__init__()
        hidden_dim = 16
        dropout = 0.2
        self.linear = nn.Sequential(
            nn.Linear(flags.class_dim, 2*hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2*hidden_dim, 2*hidden_dim),
            nn.ReLU()
        )
        
        out_dim = 48
        self.out = nn.Linear(2*hidden_dim, out_dim)
        assert not flags.factorized_representation

    def forward(self, z_style, z_content):
        x_hat = self.linear(z_content)
        x_hat = self.out(x_hat)
        return x_hat, torch.tensor(0.75).to(z_content.device)
