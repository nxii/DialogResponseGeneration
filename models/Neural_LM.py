import torch
import torch.nn as nn
import sys

class Neural_LM(nn.Module):
    
    def __init__(self, num_layers, hidden_size, bidirectional, padding_idx, embedding_matrix):
        super().__init__()
        
        self.bidirectional = bidirectional
        self.vocab_size = embedding_matrix.size(0)
        self.emb_dim = embedding_matrix.size(1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        if self.bidirectional == True:
            self.num_directions = 2
        else:
            self.num_directions = 1
          
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=padding_idx)
          
        self.lstm = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=self.bidirectional)
        
        if self.bidirectional == True:
            self.linear = nn.Linear(2*self.hidden_size, self.vocab_size)
        else:
            self.linear = nn.Linear(self.hidden_size, self.vocab_size)
    
    
    def forward(self, input_x, seq_lengths):
        
        input_x = self.embedding(input_x)
        
        packed_input_x = nn.utils.rnn.pack_padded_sequence(input_x, seq_lengths, batch_first=True)
        
        output, (h_n, c_n) = self.lstm(packed_input_x)
        
        h_n = h_n.view(self.num_layers, self.num_directions, input_x.size(0), self.hidden_size)
        h_n = h_n[-1]
        
        if self.bidirectional == True:
            h_n = torch.cat((h_n[0], h_n[1]), dim=-1)
        else:
            h_n = torch.squeeze(h_n, dim=0)
        
        padded_output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        #print(padded_output.size())
        padded_output = padded_output.view(input_x.size(0), seq_lengths[0], self.num_directions, self.hidden_size)
        #print(padded_output.size())
        if self.bidirectional == True:
            padded_output = torch.cat((padded_output[:,:,0], padded_output[:,:,1]), dim=-1)
        else:
            padded_output = torch.squeeze(padded_output, dim=-2)

        #print(padded_output.size())
        
        output = self.linear(padded_output)
        
        return h_n, output
