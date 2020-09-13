import torch
import torch.nn as nn
import torch.nn.functional as F

# Scaled dot product attention witout masking 
# building block for Multi-headed Attention Layer in Transformers
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()

        self.hidden_size = hidden_size

    def dot_product_attention(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def forward(self, hidden, encoded_output):
        energies = self.dot_product_attention(hidden, encoded_output)
        energies = energies.t()
        return F.softmax(energies, dim=1).unsqueeze(1)

class Encoder(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=2, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = embedding
        self.n_layers = n_layers
        self.hidden_size = hidden_size

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, input, input_lengths, hidden=None):
        embedded = self.embedding(input)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, enforce_sorted=False)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        #sum bidrection GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_size, embedding, output_size, n_layers=2, dropout=0.1):
        super(Decoder, self).__init__()

        self.embedding = embedding
        self.n_layers = n_layers
        self.dp = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.atten_layer = AttentionLayer(hidden_size)

    def forward(self, input, hidden, encoded_output):
        
        embedded = self.embedding(input)
        embedded = self.dp(embedded)
        # Forward through unidirectional GRU
        gru_output, hidden = self.gru(embedded, hidden)
        # Calculate attention weights from the current GRU output
        atten_weights = self.atten_layer(gru_output, encoded_output)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = atten_weights.bmm(encoded_output.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        gru_output = gru_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((gru_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, SOS_token=1):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.SOS_token = SOS_token

    def forward(self, input, input_length, max_length):
        encoded_output, encoder_hidden = self.encoder(input, input_length)
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        decoder_input = (torch.ones(1, 1, dtype=torch.long).to(self.device)) * self.SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
        all_scores = torch.zeros([0], device=self.device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoded_output)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores

def maskNLLLoss(inp, target, mask, device):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()