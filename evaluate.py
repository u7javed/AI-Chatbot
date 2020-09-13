import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np 
import time
import matplotlib.pyplot as plt
import argparse
import torch.optim as optim
from random import shuffle
import pickle

from models import *
from utilities import *

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

def load_dictionary(directory):
    with open(directory + 'dictionary.pkl', 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description='Hyperparameters for training Transformer')
    parser.add_argument('--model_directory', type=str, default='saved_models/', help='directory where models will be saved')
    
    #default hyperparameters
    parser.add_argument('--MAX_LENGTH', type=int, default=20, help='max length of a sentence')
    parser.add_argument('--epochs_trained', type=int, default=10, help='number of epochs trained in train.py')
    parser.add_argument('--hidden_size', type=int, default=512, help='size of hidden layer for encoder and decoder')
    parser.add_argument('--encoder_n_layers', type=int, default=2, help='number of encoder gru layers')
    parser.add_argument('--decoder_n_layers', type=int, default=2, help='number of decoder gru layers')
    parser.add_argument('--encoder_dropout', type=float, default=0.1, help='dropout in encoder ff')
    parser.add_argument('--decoder_dropout', type=float, default=0.1, help='dropout in decoder ff')
    
    args = parser.parse_args()
   
    model_directory = args.model_directory

    #hyperparameters
    MAX_LENGTH = args.MAX_LENGTH
    epochs_trained = args.epochs_trained
    hidden_size = args.hidden_size
    encoder_n_layers = args.encoder_n_layers
    decoder_n_layers = args.decoder_n_layers
    encoder_dropout = args.encoder_dropout
    decoder_dropout = args.decoder_dropout

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dictionary = load_dictionary(model_directory)

    #load embedder weights
    embedding = nn.Embedding(dictionary.n_count, hidden_size)
    embedding.load_state_dict(torch.load(model_directory + 'embedding_' + str(epochs_trained - 1) + '.pt'))

    #load encoder weights
    encoder = Encoder(hidden_size, embedding, encoder_n_layers, encoder_dropout).to(device)
    encoder.load_state_dict(torch.load(model_directory + 'encoder_' + str(epochs_trained - 1) + '.pt'))

    #load decoder weights
    decoder = Decoder(hidden_size, embedding, dictionary.n_count, decoder_n_layers, decoder_dropout).to(device)
    decoder.load_state_dict(torch.load(model_directory + 'decoder_' + str(epochs_trained - 1) + '.pt'))

    model = Seq2Seq(encoder, decoder, device)
    print('To exit chatbot, enter \'q\' or \'quit\'.\n')
    evaluateInput(encoder, decoder, model, dictionary, device, MAX_LENGTH)


if __name__ == "__main__":
    main()
