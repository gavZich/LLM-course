from __future__ import annotations

import os
import math
import torch
import matplotlib.pyplot as plt

import data
from transformer import TransformerLM


# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = "/content/drive/MyDrive/LM Course/Assignment-1"

DATA_PATH = os.path.join(BASE_DIR, "data/en")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints/best_model.pt")
ANALYSIS_DIR = os.path.join(BASE_DIR, "analysis")

os.makedirs(ANALYSIS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Load tokenizer and model
# --------------------------------------------------
def load_model_and_tokenizer():
    tokenizer, _ = data.load_data(DATA_PATH)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    config = checkpoint["config"]

    model = TransformerLM(
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        embed_size=config["embed_size"],
        max_context_len=config["seq_len"],
        vocab_size=config["vocab_size"],
        mlp_hidden_size=config["mlp_hidden_size"],
        with_residuals=config["with_residuals"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded checkpoint:")
    print(f"  step: {checkpoint['step']}")
    print(f"  loss: {checkpoint['loss']:.4f}")
    print(f"  device: {DEVICE}")

    return model, tokenizer, checkpoint


# --------------------------------------------------
# Run model and collect attention
# --------------------------------------------------
def collect_attention(model, tokenizer, text: str):
    token_ids = tokenizer.tokenize(text)

    if len(token_ids) > model.max_context_len:
        token_ids = token_ids[: model.max_context_len]
        text = tokenizer.detokenize(token_ids)

    x = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        _ = model(x)

    # model.last_attention_weights is a list:
    # length = num layers
    # each item shape = (B, n_heads, N, N)
    attentions = model.last_attention_weights

    print(f"Input length: {len(token_ids)}")
    print(f"Number of layers: {len(attentions)}")
    print(f"Attention tensor shape per layer: {attentions[0].shape}")

    return token_ids, text, attentions


# --------------------------------------------------
# Visualization helpers
# --------------------------------------------------
def clean_char(c: str) -> str:
    if c == "\n":
        return "\\n"
    if c == " ":
        return "␠"
    return c


def plot_attention_heatmap(att_matrix, chars, layer_idx, head_idx, title_suffix=""):
    # att_matrix shape: (N, N)
    n = len(chars)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(att_matrix, aspect="auto")

    ax.set_title(f"Attention Heatmap — Layer {layer_idx}, Head {head_idx} {title_suffix}")
    ax.set_xlabel("Key position attended to")
    ax.set_ylabel("Query position")

    labels = [clean_char(c) for c in chars]

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)

    fig.colorbar(im, ax=ax)
    plt.tight_layout()

    path = os.path.join(
        ANALYSIS_DIR,
        f"attention_heatmap_layer_{layer_idx}_head_{head_idx}.png",
    )
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()

    return path


def plot_previous_token_scores(previous_scores):
    # previous_scores is list of tuples:
    # (score, layer_idx, head_idx)
    labels = [f"L{layer}H{head}" for score, layer, head in previous_scores]
    scores = [score for score, layer, head in previous_scores]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(labels, scores)

    ax.set_title("Average Attention to Previous Token")
    ax.set_xlabel("Layer + Head")
    ax.set_ylabel("Average A[i, i-1]")
    ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()

    path = os.path.join(ANALYSIS_DIR, "previous_token_scores.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()

    return path


def plot_previous_space_scores(previous_space_scores):
    labels = [f"L{layer}H{head}" for score, layer, head in previous_space_scores]
    scores = [score for score, layer, head in previous_space_scores]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(labels, scores)

    ax.set_title("Average Attention to Previous Space")
    ax.set_xlabel("Layer + Head")
    ax.set_ylabel("Average attention to nearest previous space")
    ax.tick_params(axis="x", rotation=90)

    plt.tight_layout()

    path = os.path.join(ANALYSIS_DIR, "previous_space_scores.png")
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()

    return path


# --------------------------------------------------
# Quantitative analysis
# --------------------------------------------------
def score_previous_token(attentions):
    results = []

    for layer_idx, layer_att in enumerate(attentions):
        # layer_att shape: (B, H, N, N)
        layer_att = layer_att[0]  # (H, N, N)

        for head_idx in range(layer_att.size(0)):
            A = layer_att[head_idx]  # (N, N)
            n = A.size(0)

            if n <= 1:
                continue

            score = 0.0
            count = 0

            for i in range(1, n):
                score += A[i, i - 1].item()
                count += 1

            avg_score = score / count
            results.append((avg_score, layer_idx, head_idx))

    results.sort(reverse=True)
    return results


def score_previous_space(attentions, chars):
    results = []

    for layer_idx, layer_att in enumerate(attentions):
        layer_att = layer_att[0]  # (H, N, N)

        for head_idx in range(layer_att.size(0)):
            A = layer_att[head_idx]
            n = A.size(0)

            score = 0.0
            count = 0

            for i in range(n):
                # Find nearest previous space before position i.
                previous_space = None
                for j in range(i - 1, -1, -1):
                    if chars[j] == " ":
                        previous_space = j
                        break

                if previous_space is not None:
                    score += A[i, previous_space].item()
                    count += 1

            if count > 0:
                avg_score = score / count
                results.append((avg_score, layer_idx, head_idx))

    results.sort(reverse=True)
    return results


def print_top_results(title, results, top_k=10):
    print("")
    print(title)
    print("-" * len(title))

    for score, layer_idx, head_idx in results[:top_k]:
        print(f"Layer {layer_idx}, Head {head_idx}: {score:.4f}")


# --------------------------------------------------
# Main analysis
# --------------------------------------------------
if __name__ == "__main__":
    model, tokenizer, checkpoint = load_model_and_tokenizer()

    # You can change this text later.
    sample_text = "To be, or not to be, that is the question."

    token_ids, text, attentions = collect_attention(model, tokenizer, sample_text)
    chars = list(text)

    previous_token_scores = score_previous_token(attentions)
    previous_space_scores = score_previous_space(attentions, chars)

    print_top_results("Top heads by previous-token attention", previous_token_scores)
    print_top_results("Top heads by previous-space attention", previous_space_scores)

    plot_previous_token_scores(previous_token_scores)
    plot_previous_space_scores(previous_space_scores)

    # Plot heatmaps for the strongest discovered heads.
    best_prev_score, best_prev_layer, best_prev_head = previous_token_scores[0]
    best_space_score, best_space_layer, best_space_head = previous_space_scores[0]

    best_prev_A = attentions[best_prev_layer][0, best_prev_head].numpy()
    best_space_A = attentions[best_space_layer][0, best_space_head].numpy()

    prev_path = plot_attention_heatmap(
        best_prev_A,
        chars,
        best_prev_layer,
        best_prev_head,
        title_suffix=f"(previous-token score={best_prev_score:.3f})",
    )

    space_path = plot_attention_heatmap(
        best_space_A,
        chars,
        best_space_layer,
        best_space_head,
        title_suffix=f"(previous-space score={best_space_score:.3f})",
    )

    print("")
    print("Saved analysis files:")
    print(f"  {prev_path}")
    print(f"  {space_path}")
    print(f"  {os.path.join(ANALYSIS_DIR, 'previous_token_scores.png')}")
    print(f"  {os.path.join(ANALYSIS_DIR, 'previous_space_scores.png')}")
