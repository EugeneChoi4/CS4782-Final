import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import gensim




class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        max_seq_length: Maximum length of sequences input into the transformer.
        """
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).reshape(
            max_seq_length, 1
        )
        div_term = torch.exp(
            -1 * (torch.arange(0, d_model, 2).float() / d_model) * math.log(10000.0)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        """
        Adds the positional encoding to the model input x.
        """
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        num_heads: The number of attention heads to use.
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # TODO 9.1: define layers W_q, W_k, W_v, and W_o
        # Hint: Recall that linear layers essentially perform matrix multiplication
        #       between the layer input and layer weights
        #################

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """
        Reshapes Q, K, V into multiple heads.
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).permute(
            0, 2, 1, 3
        )

    def compute_attention(self, Q, K, V):
        """
        Returns single-headed attention between Q, K, and V.
        """
        # TODO 9.2: compute attention using the attention equation provided above
        #################
        attention = None

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        softmax = torch.softmax(scores, dim=-1)
        attention = torch.matmul(softmax, V)

        return attention

    def combine_heads(self, x):
        """
        Concatenates the outputs of each attention head into a single output.
        """
        batch_size, _, seq_length, d_k = x.size()
        return (
            x.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_length, self.d_model)
        )

    def forward(self, x):
        # TODO: 9.3 implement forward pass
        #################
        Q = self.W_q(x)  # (batch_size x seq_length x d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = self.split_heads(Q)  # (batch_size x num_heads x seq_length x d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention = self.compute_attention(Q, K, V)
        combined = self.combine_heads(attention)

        x = self.W_o(combined)

        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        d_ff: Hidden dimension size for the feed-forward network.
        """
        super(FeedForward, self).__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # TODO 10: implement feed forward pass
        #################

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, p):
        """
        Inputs:
        d_model: The dimension of the embeddings.
        num_heads: Number of heads to use for mult-head attention.
        d_ff: Hidden dimension size for the feed-forward network.
        p: Dropout probability.
        """
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p)

    def forward(self, x):

        attn_output = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class Transformer(nn.Module):
    def __init__(
        self, num_classes, d_model, num_heads, num_layers, d_ff, max_seq_length, p
    ):
        """
        Inputs:
        num_classes: Number of classes in the classification output.
        d_model: The dimension of the embeddings.
        num_heads: Number of heads to use for mult-head attention.
        num_layers: Number of encoder layers.
        d_ff: Hidden dimension size for the feed-forward network.
        max_seq_length: Maximum sequence length accepted by the transformer.
        p: Dropout probability.
        """
        super(Transformer, self).__init__()

        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(p)

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, p) for _ in range(num_layers)]
        )

        self.fc1 = nn.Linear(d_model, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):

        x = self.positional_encoding(x)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x)

        x = x.mean(dim=1)

        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


def process_batch(bert_model, data, criterion, device, val=False):
    """
    Inputs:
    data: The data in the batch to process.
    criterion: The loss function.
    val: True if processing a batch from the validation or test set.
         False if processing a batching from the training set.

    Outputs:
    Tuple of (outputs, losses)
        outputs: a dictionary containing the model outputs ('out') and predicted labels ('preds')
        metrics: a dictionary containing the model loss over the batch ('loss') and during validation (val = True),
                 the total number of examples in the batch ('batch_size') and the total number of examples whose
                 label the model predicted correctly ('num_correct')
    """

    outputs, metrics = dict(), dict()

    # TODO 13: process batch
    # Hint: For details on what information the data from the data loader contains
    #       check the __getitem__ function defined in the CustomClassDataset implemented
    #       at the beginning of Part 5
    # Hint: Make sure to send the data to the same device that the model is on.
    #################

    source_ids = data["source_ids"].to(device)
    source_mask = data["source_mask"].to(device)
    labels = data["label"].to(device)

    model_output = bert_model(input_ids=source_ids, attention_mask=source_mask)
    logits = model_output.logits 
    loss = criterion(logits, labels)
    preds = torch.argmax(logits, dim=1)

    outputs = {
        "out": logits,
        "preds": preds
    }

    metrics = {
        "loss": loss
    }

    if val:  
        batch_size = labels.size(0)
        num_correct = torch.sum(preds.eq(labels)).item()

        metrics["batch_size"] = batch_size
        metrics["num_correct"] = num_correct


    return outputs, metrics