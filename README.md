# AI-Chatbot
A Chatbot developed using Gated-Recurrent-Units(GRUs) in Pytorch. The GRU's are used in a Seq2Seq fashion which utilizes the sequential aspect of tokenized sentences and encodes them using an encoder. A decoder decodes the encoder's output and classifies them into specific tokens. A dot-product attention mechanism can be added to the decoder that learns token context by eliminating unnecessary words while keeping important ones. Teacher forcing can also be applied to create a dynamic learning method where a probability is assigned to teacher enforcing. This will allow a mixture between decoder output versus target output.

## Training
Python Files:
  - train.py
    - this is an executable python script with certain hyperparameters for training and for the models. Run this script as follows:
    ```
    python train.py
    ```
    If you want to train the default hyper paramters. If you want to add your own hyperparameters, please take a look at the script and find out the hyperparameters you can enter and then run the script as follows
    ```
    python train.py --<PARAMETER NAME HERE> <ENTER PARAMETER HERE> --<PARAMETER2 NAME HERE> <ENTER PARAMETER HERE> 
    ```
    and so on. This will save models to the saved_models directory by default as well as dictionary and embedder.
    
  - utilities.py
    - Contains various utility functions for preprocessing, training, postprocessing, and evaluating
    
  - dictionary.py
    - Contains the dictionary class that keeps track of all word/token pairs, their occurences, their index, etc.
    
  - models.py
    - Contains the models such as the decoder, encoder, attention mechanism, mask loss, and the Seq2Seq
    
  - evaluate.py
    - An executable python script. Run as follows:
    ```
    python evaluate.py
    ```
    to chat with the chatbot indefinitely or until the exit commands are entered. If you changed the hyperparamters in the trained model, please enter them according to the paramters available (refer to evaluate.py for more info.) and input them in same manner as described in the **train.py** file.
    
### To train, Run the following steps:
  - Step 1
    - Run **train.py** with or without hyperparemters (user's choice).
  - Step 2:
    - Once training is finished, run **evaluate.py** with or without its specified hyperparameters depending on whether paramters were changed during the **train.py** step. This will allow you to chat with the bot.
    
 ## Dataset
 In this project, I used the cornell movie-dialogs corpus and the specific formatted file I used is available in the data directory. The file was formatted beforehand. Thus, if you want the formatted file that this project uses, you can download or clone this repo and use the dataset file
 
 Citation: please refer to the **citation.txt** for proper referencing and citation.  

## Samples

**Input: Hello.
Bot: hello .**

**Input: It's a nice day.
Bot: yes.**

**Input: Tell me something.
Bot: i was in the middle of the place .**

**Input: Tell me a joke.
Bot: i don't care .**

**Input: Rude.
Bot: i know .**

**Input: Are you sorry?
Bot: yes .**

**Input: Goodnight.
Bot: goodnight .**
