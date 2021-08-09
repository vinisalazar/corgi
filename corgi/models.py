import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvRecurrantClassifier(nn.Module):
    def __init__(
        self,
        num_classes=None,
        num_embeddings=5,
        embedding_dim=16,
        filters=512,
        kernel_size_cnn=9,
        lstm_dims=128,
        final_layer_dims=0,  # If this is zero then it isn't used.
        dropout=0.5,
        kernel_size_maxpool=2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_embeddings = num_embeddings
        self.dropout = dropout

        ########################
        ## Embedding
        ########################
        self.embedding_dim = embedding_dim
        self.embed = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        self.dropout = nn.Dropout(dropout)

        ########################
        ## Convolutional Layer
        ########################
        self.filters = filters
        self.kernel_size_cnn = kernel_size_cnn
        self.cnn_step_1 = nn.Conv1d(
            in_channels=embedding_dim, out_channels=filters, kernel_size=kernel_size_cnn
        )
        self.max_pool_1 = nn.MaxPool1d(kernel_size=kernel_size_maxpool)

        ########################
        ## Recurrent Layer
        ########################
        self.lstm_dims = lstm_dims
        self.bi_lstm = nn.LSTM(
            input_size=filters,  # Is this dimension? - this should receive output from maxpool
            hidden_size=lstm_dims,
            bidirectional=True,
            bias=True,
            batch_first=True,
        )
        current_dims = lstm_dims * 2
        if final_layer_dims:
            self.fc1 = nn.Linear(
                in_features=current_dims,
                out_features=final_layer_dims,
            )
            current_dims = final_layer_dims

        #################################
        ## Linear Layer(s) to Predictions
        #################################
        self.final_layer_dims = final_layer_dims
        self.logits = nn.Linear(
            in_features=current_dims,
            out_features=self.num_classes,
        )

    def forward(self, x):
        ########################
        ## Embedding
        ########################
        # Cast as int because it may be simply a byte
        x = x.int()
        x = self.embed(x)

        ########################
        ## Convolutional Layer
        ########################
        x = x.transpose(
            1, 2
        )  # Transpose seq_len with embedding dims to suit convention of pytorch CNNs (batch_size, input_size, seq_len)
        x = F.relu(self.cnn_step_1(x))
        x = self.max_pool_1(x)

        ########################
        ## Recurrent Layer
        ########################

        # BiLSTM
        # Current shape: batch, filters, seq_len
        # With batch_first=True, LSTM expects shape: batch, seq, feature
        x = x.transpose(2, 1)
        output, (h_n, c_n) = self.bi_lstm(x)
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        # We are using a single layer with 2 directions so the two output vectors are
        # [0,:,:] and [1,:,:]
        # [0,:,:] -> considers the first index from the first dimension
        x = torch.cat((h_n[0, :, :], h_n[1, :, :]), dim=-1)

        #################################
        ## Linear Layer(s) to Predictions
        #################################
        # Ignore if the final_layer_dims is empty
        if self.final_layer_dims:
            x = F.relu(self.fc1(x))
        # Get logits. The cross-entropy loss optimisation function just takes in the logits and automatically does a softmax
        out = self.logits(x)

        return out
