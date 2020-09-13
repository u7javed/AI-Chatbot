# AI-Chatbot
A Chatbot developed using Gated-Recurrent-Units(GRUs) in Pytorch. This chatbot utilizes the sequential aspect of tokenized sentences and encodes them using an encoder. A decoder decodes the encoder's output and classifies them into specific tokens. A dot-product attention mechanism can be added to the decoder that learns token context by eliminating unnecessary words while keeping important ones. Teacher forcing can also be applied to create a dynamic learning method where a probability is assigned to teacher enforcing. This will allow a mixture between decoder output versus target output.

![](data/uploads/seq2seq_model.png)

## Samples

Input: Hello.
Bot: hello .

Input: It's a nice day.
Bot: yes.

Input: Tell me something.
Bot: i was in the middle of the place .

Input: Tell me a joke.
Bot: i don't care .

Input: Rude.
Bot: i know .

Input: Are you sorry?
Bot: yes .

Input: Goodnight.
Bot: goodnight .
