from torch import nn
import torch
import torch.nn.functional as F
import attention
import mlp


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        n_heads: int,
        embed_size: int,
        mlp_hidden_size: int,
        max_context_len,
        with_residuals: bool = False,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.causal_attention = attention.CausalSelfAttention(
            embed_size,
            n_heads,
            max_context_len,
        )
        self.mlp = mlp.MLP(embed_size, mlp_hidden_size)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.with_residuals = with_residuals
        self.pre_norm = pre_norm

    def forward(self, inputs):
        if self.with_residuals:
            if self.pre_norm:
                # Pre-norm: LayerNorm before each sub-layer.
                x = inputs + self.causal_attention(self.layer_norm_1(inputs))
                x = x + self.mlp(self.layer_norm_2(x))
            else:
                # Post-norm: LayerNorm after each sub-layer.
                x = self.layer_norm_1(inputs + self.causal_attention(inputs))
                x = self.layer_norm_2(x + self.mlp(x))
            return x

        x = inputs
        x = self.layer_norm_1(x)
        x = self.causal_attention(x)
        x = self.layer_norm_2(x)
        x = self.mlp(x)
        return x


class Embed(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, max_context_len):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_context_len, embed_size)
        self.max_context_len = max_context_len

    def forward(self, x):
        # x shape: (B, N), where each item is a token id.
        # output shape: (B, N, D)
        b, n = x.size()
        tok_embeddings = self.token_embeddings(x)

        device = x.device
        pos_indices = torch.arange(n, device=device)
        pos_embeddings = self.position_embeddings(pos_indices)

        return tok_embeddings + pos_embeddings


class TransformerLM(nn.Module):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        embed_size: int,
        max_context_len: int,
        vocab_size: int,
        mlp_hidden_size: int,
        with_residuals: bool,
        pre_norm: bool = True,
    ):
        super().__init__()

        self.embed = Embed(vocab_size, embed_size, max_context_len)

        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    n_heads,
                    embed_size,
                    mlp_hidden_size,
                    max_context_len,
                    with_residuals,
                    pre_norm,
                )
                for _ in range(n_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(embed_size)
        self.word_prediction = nn.Linear(embed_size, vocab_size)
        self.max_context_len = max_context_len

        # Used only for Part 5 analysis.
        # After forward, this will hold one attention tensor per layer.
        # Each tensor shape: (B, n_heads, N, N)
        self.last_attention_weights = []

        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print("Parameter count: %.2fM" % (n_params / 1e6,))

    def forward(self, inputs):
        self.last_attention_weights = []

        x = self.embed(inputs)

        for layer in self.layers:
            x = layer(x)

            # Collect attention weights from this decoder block.
            if layer.causal_attention.last_attention_weights is not None:
                self.last_attention_weights.append(
                    layer.causal_attention.last_attention_weights
                )

        x = self.layer_norm(x)
        logits = self.word_prediction(x)
        return logits

    def init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
            elif isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def sample_continuation(self, prefix: list[int], max_tokens_to_generate: int) -> list[int]:
        feed_to_lm = prefix[:]
        generated: list[int] = []

        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    feed_to_lm = feed_to_lm[-self.max_context_len:]

                device = next(self.parameters()).device
                logits = self(torch.tensor([feed_to_lm], dtype=torch.long, device=device))

                logits_for_last_token = logits[0][-1]
                distribution_for_last_token = F.softmax(logits_for_last_token, dim=-1)

                sampled_token = torch.multinomial(
                    distribution_for_last_token,
                    num_samples=1,
                ).item()

                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)

        return generated

    def better_sample_continuation(
        self,
        prefix: list[int],
        max_tokens_to_generate: int,
        temperature: float,
        topK: int,
    ) -> list[int]:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        feed_to_lm = prefix[:]
        generated: list[int] = []

        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    feed_to_lm = feed_to_lm[-self.max_context_len:]

                device = next(self.parameters()).device
                logits = self(torch.tensor([feed_to_lm], dtype=torch.long, device=device))

                logits_for_last_token = logits[0, -1] / temperature

                if topK is not None and topK > 0:
                    k = min(topK, logits_for_last_token.size(-1))
                    topk_logits, topk_indices = torch.topk(logits_for_last_token, k)

                    probs = F.softmax(topk_logits, dim=-1)
                    topk_choice = torch.multinomial(probs, num_samples=1).item()
                    sampled_token = topk_indices[topk_choice].item()
                else:
                    probs = F.softmax(logits_for_last_token, dim=-1)
                    sampled_token = torch.multinomial(probs, num_samples=1).item()

                generated.append(sampled_token)
                feed_to_lm.append(sampled_token)

        return generated