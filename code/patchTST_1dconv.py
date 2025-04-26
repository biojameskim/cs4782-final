import torch
import torch.nn as nn
from einops import rearrange

from x_transformers import Encoder
from revIN import RevIN

class Embedding(nn.Module):
    def __init__(self, input_size, embedding_dim, mode="linear"):
        super().__init__()

        self.mode = mode
        self.input_size = input_size
        self.embedding_dim = embedding_dim

        if mode == "linear":
            self.embedding = nn.Sequential(
                nn.Linear(input_size, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )
        
        if mode == "cnn":
            self.embedding = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=8, out_channels=32, kernel_size=3, padding=1),
                nn.ReLU()
            )

    def forward(self, x):
        x = x.float()

        if self.mode == "cnn":
            batch_size, num_channels, seq_len, patch_len = x.shape
            x = rearrange(x, 'b c s p -> (b c s) 1 p')
            x = self.embedding(x)
            x = rearrange(x, '(b c s) e p -> b c s (e p)', b=batch_size, c=num_channels, s=seq_len)
            return x    
        
        return self.embedding(x)
    

class PatchTSTEncoder(nn.Module):
    def __init__(self, seq_len,  num_channels, embedding_dim, heads, depth, patch_len=8, dropout=0.0, embed_mode = 'linear'):
        super().__init__()

        self.seq_len = seq_len
        self.num_channels = num_channels
        self.embed_dim = embedding_dim
        self.heads = heads
        self.depth = depth
        self.patch_len = patch_len
        self.dropout = dropout
        self.embed_mode = embed_mode

        self.embedding = Embedding(patch_len, embedding_dim, mode=self.embed_mode) # learnable embeddings for each channel
        self.pe = nn.Parameter(torch.randn(1, (seq_len // patch_len), embedding_dim)) # learnable positional encoding

        # Vanilla transformer encoder
        self.encoder = Encoder(
            dim = embedding_dim,
            depth = depth,
            heads = heads,
            dropout = self.dropout,
            sandwich_norm = True
        )

    def patchify(self, x):
        # shape x: (batch_size, seq_len, num_channels)
        x = rearrange(x, 'b (s patch_len) c -> b c s patch_len', patch_len=self.patch_len)
        return x 

    def forward(self, x):
        x = self.patchify(x)
        x = self.embedding(x)
        x = rearrange(x, 'b c num_patch emb_dim -> (b c) num_patch emb_dim') # Reshape for transformer so that channels are passed independently

        x = x + self.pe # Add positional encoding
        x = self.encoder(x)

        x = rearrange(x, '(b c) num_patch emb_dim -> b c num_patch emb_dim', c=self.num_channels)
        return x
    
class PatchTSTDecoder(nn.Module):
    def __init__(self, num_patches, num_channels, embed_dim, target_seq_size, patch_len=8, dropout=0.0):
        super().__init__()

        self.num_patches = num_patches
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.target_seq_size = target_seq_size
        self.patch_len = patch_len
        self.dropout = dropout

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(int(embed_dim * num_patches), self.target_seq_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x

class PatchTST(nn.Module):
    def __init__(self, seq_len, num_channels, embed_dim, heads, depth, target_seq_size, patch_len=8, dropout=0.0, embed_mode='linear'):
        super().__init__()

        self.encoder = PatchTSTEncoder(seq_len, num_channels, embed_dim, heads, depth, patch_len, dropout, embed_mode=embed_mode)
        self.decoder = PatchTSTDecoder(seq_len // patch_len, num_channels, embed_dim, target_seq_size, patch_len, dropout)
        self.revIN = RevIN(num_channels, affine=True, subtract_last=False)

    def forward(self, x):
        # x shape is: (batch_size, seq_len, num_channels)
        x = self.revIN(x, 'norm')
        x = self.encoder(x)
        x = self.decoder(x)

        x = rearrange(x, 'b c s -> b s c')
        x = self.revIN(x, 'denorm')

        return x # shape: (batch_size, target_seq_size, num_channels)