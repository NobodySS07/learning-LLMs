import torch
import torch.nn as nn

class TextEmbeddings(nn.Module):
    def __init__(self, vocab_size, embedding_dim, max_length, pad_idx = 0):
        super().__init__()

        self.token_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_idx)
        self.position_embeddings = nn.Embedding(num_embeddings=max_length, embedding_dim=embedding_dim, padding_idx=pad_idx)

    def forward(self, input_ids):
        batch_size, seq_length = input_ids.size()
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_length)

        token_embeds = self.token_embeddings(input_ids)
        pos_embeds = self.position_embeddings(positions)
        
        return token_embeds + pos_embeds