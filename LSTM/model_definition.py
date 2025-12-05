import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

# ==========================================
# Model Architecture (LSTM)
# ==========================================
class SensorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout=0.5):
        super(SensorLSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes) # *2 for bidirectional
        
    def forward(self, x, lengths):
        # Pack sequence to ignore padding
        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_input)
        
        # Concatenate forward and backward hidden states from the last layer
        # hidden shape: (num_layers * 2, batch, hidden_size)
        # hidden[-2]: forward of last layer, hidden[-1]: backward of last layer
        cat_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        
        out = self.dropout(cat_hidden)
        out = self.fc(out)
        return out