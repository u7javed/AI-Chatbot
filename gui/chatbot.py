from urllib import response
import utilities

class Chatbot:
    def __init__(self, encoder, decoder, seq2seq, dictionary, device, max_length):
        self.encoder = encoder
        self.decoder = decoder 
        self.seq2seq = seq2seq
        self.dictionary = dictionary
        self.device = device 
        self.max_length = max_length
        
    def response_to_input(self, input_sentence):
        response = utilities.evaluate(self.encoder, self.decoder, self.seq2seq, self.dictionary, input_sentence, self.device, self.max_length)
        response[:] = [x for x in response if not (x == 'EOS' or x == 'PAD')]
        reply = ' '.join(response)
        reply = utilities.postprocess(reply)
        return reply + '.'
