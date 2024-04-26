import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import tanh

def focal_loss(alpha, gamma, num_classes):
    # définit une fonction focal loss en fonction des paramètres et la renvoie

    def focal_loss_fixed(y_true, y_pred):
        ### Permet d'éviter les 0 et 1 avant le log ###
        epsilon = 1e-9
        y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon)

        ### Calcul de la focal loss ###
        cross_entropy = -y_true * torch.log(y_pred)
        loss = alpha * torch.pow(1 - y_pred, gamma) * cross_entropy
        return torch.mean(torch.sum(loss, dim=1))
    
    return focal_loss_fixed


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size_encoder):
        super(Encoder, self).__init__()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size_encoder)
        
    def forward(self, x): # x shape : (batch_size, seq, input_size)
        hs, cs = [], []
        for i in range(x.size(1)):
            h, c = self.lstm_cell(x[:, i, :])
            hs.append(h)
            cs.append(c)
        hs = torch.stack(hs, dim = 1)
        cs = torch.stack(cs, dim = 1)
        return hs, cs

class AttentionUnit(nn.Module):
    def __init__(self, input_size, hidden_size_encoder, v_size):
        super(AttentionUnit, self).__init__()
        self.v = torch.nn.Parameter(torch.randn(v_size, 1))
        self.W1 = nn.Linear(2*hidden_size_encoder, v_size, bias = False)
        self.W2 = nn.Linear(input_size, v_size, bias = False)
        self.tanh = nn.Tanh()

    def forward(self, x, h_previous, c_previous): # x shape (batch_size, input_size)
        alpha = torch.matmul(self.v.transpose(-2,-1), nn.Tanh()((self.W1(torch.cat((h_previous, c_previous), dim = 1)) + self.W2(x)).unsqueeze(2)))
        return alpha

class Attention(nn.Module):
    def __init__(self, input_size, hidden_size_encoder, v_size):
        super(Attention, self).__init__()
        self.decoder_unit = AttentionUnit(input_size, hidden_size_encoder, v_size)
    
    def forward(self, x, h, c):
        alpha = torch.zeros(x.size(0), x.size(1), x.size(1))
        for t in range(x.size(1)):
            for k in range(x.size(1)):
                alpha[:, t, k] = self.decoder_unit(x[:, k, :], h[:,t,:], c[:,t,:]).squeeze(2).squeeze(1)
        
        beta = nn.Softmax(dim = 2)(alpha)
        z = torch.einsum('btk,bki->bti', beta, x)
        return z
    

class LSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size_encoder, hidden_size_decoder, v_size, softmax = False):
        super(LSTMWithAttention, self).__init__()
        self.encoder = Encoder(input_size, hidden_size_encoder)
        self.attention = Attention(input_size, hidden_size_encoder, v_size)
        self.decoder = nn.LSTM(input_size, hidden_size_decoder)
        self.fc = nn.Linear(hidden_size_decoder, 2)
        self.softmax = softmax
        
    def forward(self, x):
        hs, cs = self.encoder(x)
        z = self.attention(x, hs, cs)
        z, _ = self.decoder(z)
        z = self.fc(z[:,-1,:])
        if self.softmax:
            z = F.softmax(z, dim = 1)

        return z