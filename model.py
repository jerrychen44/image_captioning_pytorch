import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        #print(resnet)#before
        modules = list(resnet.children())[:-1]
        #print(modules) #remove the finial layer fullyconnected

        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_LSTM_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_LSTM_layers = num_LSTM_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("self.embed_size",self.embed_size)
        print("self.hidden_size",self.hidden_size)
        print("self.vocab_size",self.vocab_size)
        print("self.num_LSTM_layers",self.num_LSTM_layers)
        print("self.device",self.device)

        # Embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # The LSTM takes embedded word vectors (of a specified size) as input
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(input_size=self.embed_size,
                            hidden_size=self.hidden_size, # LSTM hidden units
                            num_layers=self.num_LSTM_layers, # number of LSTM layer
                            bias=True, # use bias weights b_ih and b_hh
                            batch_first=True,  # input & output will have batch size as 1st dimension
                            dropout=0, # Not applying dropout
                            bidirectional=False, # unidirectional LSTM
                           )

        # The linear layer that maps the hidden state output dimension
        # to the number of words we want as output, vocab_size
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)



    def init_hidden(self, batch_size):
        """ At the start of training, we need to initialize a hidden state;
        there will be none because the hidden state is formed based on previously seen data.
        So, this function defines a hidden state with all zeroes
        The axes semantics are (num_LSTM_layers, batch_size, hidden_size)
        """
        #print("batch_size ",batch_size)
        return (torch.zeros((self.num_LSTM_layers, batch_size, self.hidden_size), device=self.device), \
                torch.zeros((self.num_LSTM_layers, batch_size, self.hidden_size), device=self.device))
        #return (torch.zeros((1, batch_size, self.hidden_size)),
        #        torch.zeros((1, batch_size, self.hidden_size)))

    def forward(self, features, captions):

        """ Define the feedforward behavior of the model """
        #print("captions len",(captions.shape[1]))

        # Discard the <end> word to avoid predicting when <end> is the input of the RNN
        captions = captions[:, :-1]

        # Initialize the hidden state
        batch_size = features.shape[0] # features is of shape (batch_size, embed_size)
        #print('features.shape[0]. batch_size:', features.shape[0])
        #print('features.shape[1]. embed_size:', features.shape[1])

        self.hidden = self.init_hidden(batch_size)

        # Create embedded word vectors for each word in the captions
        embeddings = self.word_embeddings(captions) # embeddings new shape : (batch_size, captions length - 1, embed_size)

        # Stack the features and captions
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1) # embeddings new shape : (batch_size, caption length, embed_size)

        # Get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hidden state
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden) # lstm_out shape : (batch_size, caption length, hidden_size)

        # Fully connected layer
        outputs = self.linear(lstm_out) # outputs shape : (batch_size, caption length, vocab_size)

        return outputs

    ## Greedy search
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "

        output = []
        batch_size = inputs.shape[0] # batch_size is 1 at inference, inputs shape : (1, 1, embed_size)
        hidden = self.init_hidden(batch_size) # Get initial hidden state of the LSTM

        while True:
            #print(inputs.shape)#torch.Size([1, 1, 512])
            #print(hidden)
            lstm_out, hidden = self.lstm(inputs, hidden) # lstm_out shape : (1, 1, hidden_size)
            outputs = self.linear(lstm_out)  # outputs shape : (1, 1, vocab_size)
            outputs = outputs.squeeze(1) # outputs shape : (1, vocab_size)
            _, max_indice = torch.max(outputs, dim=1) # predict the most likely next word, max_indice shape : (1)

            output.append(max_indice.cpu().numpy()[0].item()) # storing the word predicted

            if (max_indice == 1):
                # We predicted the <end> word, so there is no further prediction to do
                break

            ## Prepare to embed the last predicted word to be the new input of the lstm
            #  Embedding layer that turns words into a vector of a specified size
            #  prepare the inputs for next time t+1
            inputs = self.word_embeddings(max_indice) # inputs shape : (1, embed_size)
            inputs = inputs.unsqueeze(1) # inputs shape : (1, 1, embed_size)

        return output