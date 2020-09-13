import torch
import numpy as np 
import argparse
from random import shuffle
import pickle

from models import *
from utilities import *

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

class Trainer():

    def save_dictionary(self, dictionary):
        with open(self.saved_model_directory + 'dictionary.pkl', 'wb') as f:
            pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)

    def __init__(self, file_directory, saved_model_directory, batch_size=128, hidden_size=512, encoder_n_layers=2, decoder_n_layers=2, 
                encoder_dropout=0.1, decoder_dropout=0.1, encoder_lr=0.0001, decoder_lr=0.0005, tf_ratio=1.0, 
                clip=50.0, MAX_LENGTH=20, device='cpu'):

        self.tf_ratio = tf_ratio
        self.saved_model_directory = saved_model_directory
        self.device = device
        self.clip = clip

        pairs = []
        #split into pairs
        for line in open(file_directory, 'r').readlines():
            pair = line.split('\t')
            pairs.append([pair[0], pair[1]])

        #preprocess pairs
        for pair in pairs:
            pair[0] = preprocess(pair[0].lower())
            pair[1] = preprocess(pair[1].lower())
        
        #trim pairs by MAX_LENGTH
        trimmed_pairs = []
        for pair in pairs:
            if len(pair[0].split(' ')) <= MAX_LENGTH and len(pair[1].split(' ')) <= MAX_LENGTH:
                trimmed_pairs.append(pair)

        #define dictionary
        dictionary = Dictionary('corpus')
        for pair in trimmed_pairs:
            dictionary.add_sentence(pair[0])
            dictionary.add_sentence(pair[1])
        
        #save dictionary
        self.save_dictionary(dictionary)

        #tokenized pairs
        tokenized_pairs = []
        for pair in trimmed_pairs:
            tokenized_pairs.append([tokenize(pair[0], dictionary, MAX_LENGTH), tokenize(pair[1], dictionary, MAX_LENGTH)])
        #convert to numpy array
        tokenized_pairs = np.array(tokenized_pairs)

        self.data_loader = create_batches(tokenized_pairs, batch_size, self.device)

        self.embedding = nn.Embedding(dictionary.n_count, hidden_size)
        self.encoder = Encoder(hidden_size, self.embedding, n_layers=encoder_n_layers, dropout=encoder_dropout).to(self.device)
        self.decoder = Decoder(hidden_size, self.embedding, dictionary.n_count, n_layers=decoder_n_layers, dropout=decoder_dropout).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=encoder_lr)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=decoder_lr)
        

    def train(self, epochs):
        for epoch in range(epochs):
            shuffle(self.data_loader)
            for i, (inp, lengths, target, mask, max_target_len) in enumerate(self.data_loader):
                b_size = inp.size(1)
                loss = 0
                n_totals = 0

                self.encoder_optimizer.zero_grad()
                self.decoder_optimizer.zero_grad()

                encoder_outputs, encoder_hidden = self.encoder(inp, lengths)

                decoder_inp = torch.LongTensor([[SOS_TOKEN for _ in range(b_size)]])
                decoder_inp = decoder_inp.to(self.device)

                decoder_hidden = encoder_hidden[:self.decoder.n_layers]

                teacher_forcing = True if random.random() < self.tf_ratio else False
                if teacher_forcing:
                    for t in range(max_target_len):
                        decoder_output, decoder_hidden = self.decoder(decoder_inp, decoder_hidden, encoder_outputs)
                        decoder_inp = target[t].view(1, -1)
                        mask_loss, nTotal = maskNLLLoss(decoder_output, target[t], mask[t], self.device)
                        loss += mask_loss
                        n_totals += nTotal
                else:
                    for t in range(max_target_len):
                        decoder_output, decoder_hidden = self.decoder(decoder_inp, decoder_hidden, encoder_outputs)

                        _, topi = decoder_output.topk(1)
                        decoder_inp = torch.LongTensor([[topi[i][0] for i in range(b_size)]])
                        decoder_inp = decoder_inp.to(self.device)

                        mask_loss, nTotal = maskNLLLoss(decoder_output, target[t], mask[t], self.device)
                        loss += mask_loss
                        n_totals += nTotal

                loss.backward()

                _ = nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
                _ = nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)

                self.encoder_optimizer.step()
                self.decoder_optimizer.step()

                if i % 100 == 0:
                    print('[{}/{}][{}/{}],  Training Loss: {:.4f}'.format(epoch, epochs, i, len(self.data_loader), loss.item()*100/n_totals))

            #print loss
            print('Epoch [{}/{}],   Training Loss: {:.4f}'.format(epoch, epochs, loss.item()*100/n_totals))

            #save models
            torch.save(self.encoder.state_dict(), self.saved_model_directory + 'encoder_{}.pt'.format(epoch))
            torch.save(self.decoder.state_dict(),  self.saved_model_directory + 'decoder_{}.pt'.format(epoch))
            torch.save(self.embedding.state_dict(),  self.saved_model_directory + 'embedding_{}.pt'.format(epoch))
        print('Training Finished')

def main():
    parser = argparse.ArgumentParser(description='Hyperparameters for training Transformer')
    #hyperparameter loading
    parser.add_argument('--file_directory', type=str, default='data/formatted_movie_lines.txt', help='location of formatted file')
    parser.add_argument('--saved_model_directory', type=str, default='saved_models/', help='directory where models will be saved')
    parser.add_argument('--batch_size', type=int, default=128, help='size of batches per feedthrough')
    parser.add_argument('--epochs', type=int, default=10, help='number of training iterations through entire dataset')
    parser.add_argument('--hidden_size', type=int, default=512, help='size of hidden layer for encoder and decoder')
    parser.add_argument('--encoder_n_layers', type=int, default=2, help='number of encoder gru layers')
    parser.add_argument('--decoder_n_layers', type=int, default=2, help='number of decoder gru layers')
    parser.add_argument('--encoder_dropout', type=float, default=0.1, help='dropout in encoder ff')
    parser.add_argument('--decoder_dropout', type=float, default=0.1, help='dropout in decoder ff')
    parser.add_argument('--encoder_lr', type=float, default=0.0001, help='learning rate for encoder')
    parser.add_argument('--decoder_lr', type=float, default=0.0005, help='learning rate for decoder')
    parser.add_argument('--tf_ratio', type=float, default=0.1, help='probability of using teacher forcing')
    parser.add_argument('--clip', type=float, default=50.0, help='cliping size for gradient norm')
    parser.add_argument('--MAX_LENGTH', type=int, default=20, help='max length of sentences')
    parser.add_argument('--device', type=str, default='cpu', help='device to run computations')
    
    args = parser.parse_args()
   
    file_directory = args.file_directory
    saved_model_directory = args.saved_model_directory
    batch_size = args.batch_size
    epochs = args.epochs
    hidden_size = args.hidden_size
    encoder_n_layers = args.encoder_n_layers
    decoder_n_layers = args.decoder_n_layers
    encoder_dropout = args.encoder_dropout
    decoder_dropout = args.decoder_dropout
    encoder_lr = args.encoder_lr
    decoder_lr = args.decoder_lr
    tf_ratio = args.tf_ratio
    clip = args.clip
    MAX_LENGTH = args.MAX_LENGTH
    device = args.device

    model = Trainer(file_directory, saved_model_directory, batch_size, hidden_size, encoder_n_layers,
                    decoder_n_layers, encoder_dropout, decoder_dropout, encoder_lr, decoder_lr, tf_ratio,
                    clip, MAX_LENGTH, device)
    model.train(epochs)

if __name__ == "__main__":
    main()
