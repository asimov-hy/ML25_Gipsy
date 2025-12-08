import torch
import torch.nn as nn

class SensorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.4):
        """
        Bidirectional LSTM Model.
        dropout_rate: Increased to 0.4 to prevent overfitting.
        """
        super(SensorLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Fully Connected Layer
        # input dim = hidden_size * 2 (because of bidirectional)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
        # Dropout layer for the final classification head
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, lengths):
        # Pack the sequence
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=True)
        
        # Forward pass through LSTM
        packed_output, (hidden, cell) = self.lstm(packed_input)
        
        # Concatenate the final forward and backward hidden states
        # hidden shape: (num_layers * num_directions, batch, hidden_size)
        
        # Take the last layer's hidden state
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        
        # Shape: (batch, hidden_size * 2)
        cat_hidden = torch.cat((hidden_forward, hidden_backward), dim=1)
        
        # Apply Dropout
        out = self.dropout(cat_hidden)
        
        # Classification
        out = self.fc(out)
        
        return out