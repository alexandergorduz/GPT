import torch
import torch.nn as nn



"""
S - sequences dimension
T - tokens dimension
C - embeddings channels dimension
QC - head query channels dimension
KC - head key channels dimension
VC - head value channels dimension
H - heads dimension
V - vocabulary dimension
"""



class Embedding(nn.Module):
    """
    Class Embedding for creating and using tokens embeddings.
    """


    def __init__(self, vocab_size: int, d_model: int) -> None:
        """
        Initialize the Embedding class.

        Args:
            vocab_size (int): size of the vocabulary.
            d_model (int): dimentionality of the embedding vector.
        """

        super(Embedding, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the embedding layer.

        Args:
            x (torch.Tensor): input tensor with tokens indices.

        Returns:
            torch.Tensor: output tensor with tokens embeddings.
        """

        output = self.embedding(x) * self.d_model ** 0.5 # (S, T, C)

        return output



class PositionEncoder(nn.Module):
    """
    Class PositionEncoder for adding position encoding to tokens embeddings.
    """


    def __init__(self, seq_size: int, d_model: int) -> None:
        """
        Initialize the PositionEncoder class.

        Args:
            seq_size (int): maximum sequence length.
            d_model (int): dimentionality of the embedding vector.
        """

        super(PositionEncoder, self).__init__()

        pos_enc = torch.zeros(seq_size, d_model)
        pos = torch.arange(0, seq_size, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div)
        pos_enc[:, 1::2] = torch.cos(pos * div)
        pos_enc = pos_enc.unsqueeze(0)

        self.register_buffer("pos_enc", pos_enc)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs an adding positional encoding to the input tokens embeddings.

        Args:
            x (torch.Tensor): input tensor with tokens embeddings.

        Returns:
            torch.Tensor: output tensor with tokens embeddings and positional encoding.
        """

        tokens_amt = x.shape[1]

        output = x + self.pos_enc[:, :tokens_amt] # (S, T, C)

        return output



class Head(nn.Module):
    """
    Class Head for applying attention mechanism for one head.
    """


    def __init__(self, d_model: int,
                 d_query: int, d_key: int, d_value: int,
                 dropout: float) -> None:
        """
        Initialize the Head class.

        Args:
            d_model (int): dimentionality of the embedding vector.
            d_query (int): dimentionality of the head query vector.
            d_key (int): dimentionality of the head key vector.
            d_value (int): dimentionality of the head value vector.
            dropout (float): dropout probability.
        """

        super(Head, self).__init__()

        self.W_query = nn.Linear(d_model, d_query, bias=False)
        self.W_key = nn.Linear(d_model, d_key, bias=False)
        self.W_value = nn.Linear(d_model, d_value, bias=False)
        self.dropout = nn.Dropout(dropout)


    def forward(self,
                q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Performs a transforming input tokens embeddings according to attention mechanism for one head.

        Args:
            q (torch.Tensor): input query tensor with tokens embeddings.
            k (torch.Tensor): input key tensor with tokens embeddings.
            v (torch.Tensor): input value tensor with tokens embeddings.
            mask (torch.Tensor): mask tensor to perform masking mechanism.

        Returns:
            torch.Tensor: output tensor with tokens embeddings and applied attention mechanism for one head.
        """

        tokens_amt = q.shape[1]

        query = self.W_query(q) # (S, T, QC)
        key = self.W_key(k) # (S, T, KC)
        value = self.W_value(v) # (S, T, VC)

        weights = torch.matmul(query, key.transpose(-2, -1)) / query.shape[-1] ** 0.5 # (S, T, QC) @ (S, T, KC) --> (S, T, T)

        if mask is not None:

            weights = weights.masked_fill(mask[:tokens_amt, :tokens_amt] == 0, float("-inf")) # (S, T, T)

        weights = torch.softmax(weights, dim=-1) # (S, T, T)

        output = torch.matmul(weights, value) # (S, T, T) @ (S, T, VC) --> (S, T, VC)

        output = self.dropout(output) # (S, T, VC)

        return output



class MultiHead(nn.Module):
    """
    Class MultiHead for applying multiple attention heads.
    """


    def __init__(self, heads_amt: int, seq_size: int,
                 d_model: int, d_query: int, d_key: int, d_value: int,
                 dropout: float, is_masked: bool = False) -> None:
        """
        Initialize the MultiHead class.

        Args:
            heads_amt (int): number of attention heads.
            seq_size (int): maximum sequence length.
            d_model (int): dimentionality of the embedding vector.
            d_query (int): dimentionality of the head query vector.
            d_key (int): dimentionality of the head key vector.
            d_value (int): dimentionality of the head value vector.
            dropout (float): dropout probability.
            is_masked (bool): whether the multihead is masked.
        """

        super(MultiHead, self).__init__()

        self.heads = nn.ModuleList([Head(d_model, d_query, d_key, d_value, dropout) for _ in range(heads_amt)])

        if is_masked:

            mask = torch.tril(torch.ones(seq_size, seq_size))

            self.register_buffer("mask", mask)

        self.W_output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.is_masked = is_masked


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Performs a transforming input tokens embeddings according to attention mechanism for multihead.

        Args:
            q (torch.Tensor): input query tensor with tokens embeddings.
            k (torch.Tensor): input key tensor with tokens embeddings.
            v (torch.Tensor): input value tensor with tokens embeddings.

        Returns:
            torch.Tensor: output tensor with tokens embeddings and applied attention mechanism for one multihead.
        """

        if self.is_masked:

            output = torch.cat([head(q, k, v, self.mask) for head in self.heads], dim=-1) # (S, T, VC, H) --> (S, T, C)

        else:

            output = torch.cat([head(q, k, v) for head in self.heads], dim=-1) # (S, T, VC, H) --> (S, T, C)

        output = self.W_output(output) # (S, T, C)

        output = self.dropout(output) # (S, T, C)

        return output



class FeedForward(nn.Module):
    """
    Class FeedForward for creating and using feed-forward neural network.
    """


    def __init__(self, d_model: int, d_feed_forward: int,
                 dropout: float) -> None:
        """
        Initialize the FeedForward class.

        Args:
            d_model (int): dimentionality of the embedding vector.
            d_feed_forward (int): dimentionality of the neural network hidden layer.
            dropout (float): dropout probability.
        """

        super(FeedForward, self).__init__()

        self.fully_conn_1 = nn.Sequential(
            nn.Linear(d_model, d_feed_forward),
            nn.ReLU()
        )
        self.fully_conn_2 = nn.Linear(d_feed_forward, d_model)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward propagation through feed-forward neural network.

        Args:
            x (torch.Tensor): input tensor with tokens embeddings.

        Returns:
            torch.Tensor: output tensor with tokens embeddings after propagation through feed-forward neural network.
        """

        output = self.fully_conn_1(x) # (S, T, C)

        output = self.dropout(output) # (S, T, C)

        output = self.fully_conn_2(output) # (S, T, C)

        return output



class LayerNorm(nn.Module):
    """
    Class LayerNorm for applying layer normalization mechanism.
    """


    def __init__(self, d_model: int,
                 eps: float = 1e-6) -> None:
        """
        Initialize the LayerNorm class.

        Args:
            d_model (int): dimentionality of the embedding vector.
            eps (float): prevention dividing by zero.
        """

        super(LayerNorm, self).__init__()

        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward propagation through layer normalization.

        Args:
            x (torch.Tensor): input tensor with tokens embeddings.

        Returns:
            torch.Tensor: output tensor with tokens embeddings after layer normalization.
        """

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        x_norm = (x - mean) / (std + self.eps) # (S, T, C)

        output = self.gamma * x_norm + self.beta # (S, T, C)

        return output



class DecoderLayer(nn.Module):
    """
    Class DecoderLayer for creating and using one MultiHead-FeedForward decoder layer.
    """


    def __init__(self, heads_amt: int, seq_size: int,
                 d_model: int, d_query: int, d_key: int,
                 d_value: int, d_feed_forward: int,
                 dropout: float) -> None:
        """
        Initialize the DecoderLayer class.

        Args:
            heads_amt (int): number of attention heads.
            seq_size (int): maximum sequence length.
            d_model (int): dimentionality of the embedding vector.
            d_query (int): dimentionality of the head query vector.
            d_key (int): dimentionality of the head key vector.
            d_value (int): dimentionality of the head value vector.
            d_feed_forward (int): dimentionality of the neural network hidden layer.
            dropout (float): dropout probability.
        """

        super(DecoderLayer, self).__init__()

        self.multihead = MultiHead(heads_amt, seq_size, d_model, d_query, d_key, d_value, dropout, is_masked=True)
        self.multihead_layernorm = LayerNorm(d_model)
        self.feedforward = FeedForward(d_model, d_feed_forward, dropout)
        self.feedforward_layernorm = LayerNorm(d_model)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward propagation through MultiHead-FeedForward decoder layer.

        Args:
            x (torch.Tensor): input tensor with tokens embeddings.

        Returns:
            torch.Tensor: output tensor with tokens embeddings after propagation through MultiHead-FeedForward decoder layer.
        """

        multihead_output = self.multihead(x, x, x) # (S, T, C)
        multihead_output = multihead_output + x # (S, T, C)
        multihead_output = self.multihead_layernorm(multihead_output) # (S, T, C)
        feedforward_output = self.feedforward(multihead_output) # (S, T, C)
        feedforward_output = feedforward_output + multihead_output # (S, T, C)

        output = self.feedforward_layernorm(feedforward_output) # (S, T, C)

        return output



class GPT(nn.Module):
    """
    Class GPT for creating and using generative pre-trained transformer.
    """


    def __init__(self, vocab_size: int, seq_size: int,
                 heads_amt: int, layers_amt: int,
                 d_model: int, d_query: int, d_key: int,
                 d_value: int, d_feed_forward: int,
                 dropout: float) -> None:
        """
        Initialize the GPT class.

        Args:
            vocab_size (int): size of the vocabulary.
            seq_size (int): maximum sequence length.
            heads_amt (int): number of attention heads.
            layers_amt (int): number of decoder layers.
            d_model (int): dimentionality of the embedding vector.
            d_query (int): dimentionality of the head query vector.
            d_key (int): dimentionality of the head key vector.
            d_value (int): dimentionality of the head value vector.
            d_feed_forward (int): dimentionality of the neural network hidden layer.
            dropout (float): dropout probability.
        """

        super(GPT, self).__init__()

        self.seq_size = seq_size
        self.embedding = Embedding(vocab_size, d_model)
        self.position_encoder = PositionEncoder(seq_size, d_model)
        self.decoder_layers = nn.Sequential(*[DecoderLayer(heads_amt, seq_size, d_model, d_query, d_key, d_value, d_feed_forward, dropout) for _ in range(layers_amt)])
        self.linear = nn.Linear(d_model, vocab_size)

        self.apply(self._params_init)


    def _params_init(self, module: torch.nn.Module) -> None:
        """
        Performs a best model parameters initialization.

        Args:
            module (torch.nn.Module): module for parameters initialization.
        """

        if isinstance(module, nn.Linear):

            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:

                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):

            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward propagation through generative pre-trained transformer.

        Args:
            x (torch.Tensor): input tensor with tokens indices.

        Returns:
            torch.Tensor: output tensor with tokens logits.
        """

        output = self.embedding(x) # (S, T, C)
        output = self.position_encoder(output) # (S, T, C)
        output = self.decoder_layers(output) # (S, T, C)
        output = self.linear(output) # (S, T, V)

        return output


    @torch.no_grad()
    def generate(self, x: torch.Tensor, tokens_amt: int) -> torch.Tensor:
        """
        Performs a tokens sequence generation.

        Args:
            x (torch.Tensor): input tensor with tokens indices as a context to generate sequence.
            tokens_amt (int): amount of tokens to generate.

        Returns:
            torch.Tensor: output tensor with tokens indices as an answer sequence.
        """

        for _ in range(tokens_amt):

            x_prev = x[:, -self.seq_size:] # (S, T)
            logits = self(x_prev) # (S, T, V)
            logits = logits[:, -1, :] # (S, V)
            probas = torch.softmax(logits, dim=-1) # (S, V)
            x_next = torch.multinomial(probas, num_samples=1) # (S, T)
            x = torch.cat([x, x_next], dim=1) # (S, T)

        return x