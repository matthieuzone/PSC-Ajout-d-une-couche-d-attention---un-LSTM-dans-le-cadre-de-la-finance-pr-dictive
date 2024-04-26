import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.functional import tanh
def focal_loss(alpha, gamma):
    # définit une fonction focal loss en fonction des paramètres et la renvoie
    def focal_loss_fixed(y_true, y_pred):
        
        epsilon = 1e-9
        y_pred = torch.clamp(y_pred, epsilon, 1. - epsilon) # Permet d'éviter les 0 et 1 avant le log

        ### Calcul de la focal loss ###
        cross_entropy = -y_true * torch.log(y_pred)
        alpha_ = torch.tensor([alpha,1-alpha])
        loss = alpha_ * torch.pow(1 - y_pred, gamma) * cross_entropy
        return torch.sum(loss)
    return focal_loss_fixed


class Encoder(nn.Module): # Encodeur pour le LSTM avec attention
    def _init_(self, input_size, hidden_size_encoder):
        super(Encoder, self)._init_()
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size_encoder) # Cellule LSTM. Nous n'utilisons pas nn.LSTM car nous voulons le cell state et le hidden state à chaque étape
        
    def forward(self, x): # x shape (batch_size, seq, input_size)
        hs, cs = [], [] # Liste pour stocker les hidden states et cell states
        for i in range(x.size(1)):
            h, c = self.lstm_cell(x[:, i, :]) # On passe chaque élément de la séquence à la cellule LSTM
            hs.append(h) 
            cs.append(c)
        hs = torch.stack(hs, dim = 1)
        cs = torch.stack(cs, dim = 1) 
        return hs, cs # On retourne les hidden states et cell states

class AttentionUnit(nn.Module): # Unité d'attention
    def _init_(self, input_size, hidden_size_encoder, v_size):
        super(AttentionUnit, self)._init_()
        self.v = torch.nn.Parameter(torch.randn(v_size, 1)) # Paramètre v
        self.W1 = nn.Linear(2*hidden_size_encoder, v_size, bias = False) # Poids W1
        self.W2 = nn.Linear(input_size, v_size, bias = False) # Poids W2
        self.tanh = nn.Tanh()

    def forward(self, x, h_previous, c_previous): # x shape (batch_size, input_size)
        alpha = torch.matmul(self.v.transpose(-2,-1), nn.Tanh()((self.W1(torch.cat((h_previous, c_previous), dim = 1)) + self.W2(x)).unsqueeze(2))) # Calcul d'un scalaire alpha
        return alpha

class Attention(nn.Module):
    def _init_(self, input_size, hidden_size_encoder, v_size):
        super(Attention, self)._init_()
        self.decoder_unit = AttentionUnit(input_size, hidden_size_encoder, v_size)
    
    def forward(self, x, h, c):
        alpha = torch.zeros(x.size(0), x.size(1), x.size(1))
        for t in range(x.size(1)):
            for k in range(x.size(1)):
                alpha[:, t, k] = self.decoder_unit(x[:, k, :], h[:,t,:], c[:,t,:]).squeeze(2).squeeze(1) # On calcule les alphas pour chaque t et k de la séquence
        
        beta = nn.Softmax(dim = 2)(alpha)
        z = torch.einsum('btk,bki->bti', beta, x) # On calcule z en faisant une somme pondérée des éléments de la séquence par les betas
        return z
    

class LSTMWithAttention(nn.Module):
    def _init_(self, input_size, hidden_size_encoder, hidden_size_decoder, v_size, softmax = False):
        super(LSTMWithAttention, self)._init_()
        self.encoder = Encoder(input_size, hidden_size_encoder) # Encodeur LSTM
        self.attention = Attention(input_size, hidden_size_encoder, v_size) #  Couche d'attention
        self.decoder = nn.LSTM(input_size, hidden_size_decoder) # Décodeur LSTM
        self.fc = nn.Linear(hidden_size_decoder, 2) # Couche linéaire pour la classification
        self.softmax = softmax
        
    def forward(self, x):
        hs, cs = self.encoder(x) # On récupère les hidden states et cell states
        z = self.attention(x, hs, cs) # On calcule z, la nouvelle séquence résultat de la somme pondérée par l'attention de la séquence
        z, _ = self.decoder(z) # On passe z dans le décodeur LSTM
        z = self.fc(z[:,-1,:]) # On passe le dernier élément de la séquence dans une couche linéaire
        if self.softmax:
            z = F.softmax(z, dim = 1)

        return z