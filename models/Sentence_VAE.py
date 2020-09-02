import torch
import torch.nn as nn
import sys

class Sentence_VAE(nn.Module):
    
    def __init__(self, encoder_layers, decoder_layers, hidden_size, latent_size, bidirectional, temperature, max_seq_len, dropout, pad_idx, sos_idx, eos_idx, embedding_matrix):
        super().__init__()
        
        self.pad_idx = pad_idx
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        
        self.max_seq_len = max_seq_len
        
        self.bidirectional = bidirectional
        self.temperature = temperature
        self.vocab_size = embedding_matrix.size(0)
        self.emb_dim = embedding_matrix.size(1)
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.decoder_layers = decoder_layers
        self.encoder_layers = encoder_layers
        
        if self.bidirectional == True:
            self.num_directions = 2
        else:
            self.num_directions = 1
          
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=self.pad_idx)
          
        self.encoder = nn.LSTM(input_size=self.emb_dim, hidden_size=self.hidden_size, num_layers=self.encoder_layers, batch_first=True, bidirectional=self.bidirectional, dropout=dropout)
        
        self.hidden2mean = nn.Linear(self.encoder_layers*self.num_directions*self.hidden_size, self.latent_size)
        self.hidden2logv = nn.Linear(self.encoder_layers*self.num_directions*self.hidden_size, self.latent_size)
        
        self.decoder = nn.LSTM(input_size=self.emb_dim+self.latent_size, hidden_size=self.hidden_size, num_layers=self.decoder_layers, batch_first=True, bidirectional=False, dropout=dropout)
        
        self.outputs2vocab = nn.Linear(self.hidden_size, self.vocab_size)
        
    
    def encode(self, packed_input_x, batch_size, max_seq_len):
        
        output, (h_n, c_n) = self.encoder(packed_input_x)
        
        padded_output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        h_n = h_n.view(self.encoder_layers, self.num_directions, batch_size, self.hidden_size)
        
        if self.bidirectional == True:
            hidden = torch.cat((h_n[:,0], h_n[:,1]), dim=-1)
        else:
            hidden = torch.squeeze(h_n, dim=1)
            
        
        if self.encoder_layers > 1:
            hidden_seq = []
            for index in range(self.encoder_layers):
                hidden_seq.append(hidden[index])
            
            hidden = torch.cat(hidden_seq, dim=-1)
        else:
            hidden = torch.squeeze(torch.squeeze(h_n, dim=0))
            
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        
        # Reparameterization trick
        eps = torch.randn(mean.size()).cuda()
        z = mean + (self.temperature * (eps * logv.exp()))
        
        return h_n, mean, logv, z
    
    def decode(self, packed_decoder_input):
        
        output, _ = self.decoder(packed_decoder_input)
        
        padded_output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        logits = self.outputs2vocab(padded_output)
        
        return logits
    
    
    def inference(self, z=None, samples=None):
        
        if z == None:
            z = torch.randn(samples, self.latent_size).cuda()
        
        batch_size = z.size(0)
        
        z = z.unsqueeze(dim=1)
        
        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size).long().cuda()  # all idx of batch
        # all idx of batch which are still generating
        sequence_running = torch.arange(0, batch_size).long().cuda()
        sequence_mask = torch.ones(batch_size).bool().cuda()
        # idx of still generating sequences with respect to current loop
        running_seqs = torch.arange(0, batch_size).long().cuda()
        
        generations = torch.zeros(batch_size, self.max_seq_len).fill_(self.pad_idx).long().cuda()

        t = 0
        while t < self.max_seq_len and len(running_seqs) > 0:

            if t == 0:
                input_sequence = torch.zeros(batch_size).fill_(self.sos_idx).long().cuda()
            
            input_sequence = input_sequence.unsqueeze(-1)

            input_embedding = self.embedding(input_sequence)
            
            decoder_input = torch.cat((input_embedding, z), dim=-1)

            if t == 0:
                output, (h_n, c_n) = self.decoder(decoder_input)
            else:
                output, (h_n, c_n) = self.decoder(decoder_input, (h_n, c_n))

            logits = self.outputs2vocab(output)
            
            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update global running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx)
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence[running_seqs]
                z = z[running_seqs]
                h_n = h_n[:, running_seqs]
                c_n = c_n[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs)).long().cuda()

            t += 1

        return generations
        
    
    def _sample(self, dist, mode='greedy'):
        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
            
        sample = sample.squeeze()
        
        if dist.size(0) == 1:
            sample = sample.unsqueeze(0)

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to
    
    
    def forward(self, input_x, seq_lengths, inference=False):
        
        input_x = self.embedding(input_x)
        
        packed_input_x = nn.utils.rnn.pack_padded_sequence(input_x, seq_lengths, batch_first=True)
        
        h_n, mean, logv, z = self.encode(packed_input_x, input_x.size(0), seq_lengths[0])
        
        if inference == False:
            decoder_input = input_x.clone().detach()
            stacked_z = torch.stack([z]*30, dim=1)
            decoder_input = torch.cat((decoder_input, stacked_z), dim=-1)
            
            packed_decoder_input = nn.utils.rnn.pack_padded_sequence(decoder_input, seq_lengths, batch_first=True)
            
            logits = self.decode(packed_decoder_input)
            
            return h_n, mean, logv, z, logits
        else:
            generated_seqs = self.inference(z)
            
            return h_n, mean, logv, z, generated_seqs
            
        
        
