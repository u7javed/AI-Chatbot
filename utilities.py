import re
from dictionary import Dictionary
import torch
import random
from random import shuffle
import numpy as np 

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

def preprocess(s):
    s = s.replace('\n', '')
    s = s.replace('...', '.')
    s = re.sub(r"([.!?])", r" \1", s)
    s = s.replace('\'', '')
    s = s.replace(',', '')
    s = s.replace('-', ' ')
    s = re.sub(' +', ' ', s)
    return s

def tokenize(sentence, dictionary, MAX_LENGTH):
    split_sentence = [word for word in sentence.split(' ')]
    token = [dictionary.word2index[word] for word in sentence.split(' ')]
    token.append(EOS_TOKEN)
    token += [PAD_TOKEN]*(MAX_LENGTH - len(split_sentence))
    return token

def create_batches(pairs, batch_size, device):
    data_loader = []
    for i in range(0, len(pairs), batch_size):
        seq_length = min(len(pairs) - i, batch_size)

        input_batch = pairs[i:i+seq_length, 0, :]
        target_batch = pairs[i:i+seq_length, 1, :]

        lengths = torch.LongTensor([len([token for token in s if token != 0]) for s in input_batch]).to(device)
        max_target_len = max([len([token for token in s if token != 0]) for s in target_batch])

        input_tensor = torch.LongTensor(input_batch).t().to(device)
        target_tensor = torch.LongTensor(target_batch).t().to(device)

        mask = target_tensor != PAD_TOKEN
        data_loader.append([input_tensor, lengths, target_tensor, mask, max_target_len])

    return data_loader

def evaluate(encoder, decoder, seq2seq, dictionary, sentence, device, max_length):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [tokenize(sentence.lower(), dictionary, max_length)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with seq2seq
    tokens, scores = seq2seq(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [dictionary.index2word[token.item()] for token in tokens]
    return decoded_words

def postprocess(sentence):
    spliter = sentence.split('.')
    return spliter[0]


def evaluateInput(encoder, decoder, seq2seq, dictionary, device, max_length):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('>')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            preprocessed_sentence = preprocess(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, seq2seq, dictionary, preprocessed_sentence, device, max_length)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            reply = ' '.join(output_words)
            reply = postprocess(reply)
            reply = reply + '.'
            print('You: ', input_sentence)
            print('Bot:', reply)

        except KeyError:
            print("Error: Encountered unknown word.")