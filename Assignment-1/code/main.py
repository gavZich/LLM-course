from __future__ import annotations

import torch

if __name__ == "__main__":
    import lm
    import torch
    import math
    from torch import nn, optim
    from transformer import TransformerLM
    import matplotlib.pyplot as plt
    from IPython.display import display, clear_output

    import data

    seq_len = 128
    batch_size = 64
    data_path = "/content/drive/MyDrive/LM Course/Assignment1/en"
    n_layers = 6
    n_heads = 6
    embed_size = 192
    mlp_hidden_size = embed_size * 4

    learning_rate = 5e-4
    gradient_clipping = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_batches_to_train = 50000
    plot_every = 500

    tokenizer, tokenized_data = data.load_data(data_path)
    # NOTE: are data items are longer by one than the sequence length,
    # They will be shortened by 1 when converted to training examples.
    data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))

    model: torch.nn.Module = TransformerLM(
        n_layers,
        n_heads,
        embed_size,
        seq_len,
        tokenizer.vocab_size(),
        mlp_hidden_size,
        with_residuals=True,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=[0.9, 0.95])

    # ---- Tracking lists ----
    steps_log = []
    loss_log = []
    grad_norm_log = []
    weight_norm_log = []
    adam_momentum_log = []
    adam_variance_log = []
    perplexity_log = []

    def compute_grad_norm(model):
        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return math.sqrt(total)

    def compute_weight_norm(model):
        total = 0.0
        for p in model.parameters():
            total += p.data.norm(2).item() ** 2
        return math.sqrt(total)

    def compute_adam_stats(optimizer):
        momentum_avg, variance_avg, count = 0.0, 0.0, 0
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state.get(p)
                if state and 'exp_avg' in state:
                    momentum_avg += state['exp_avg'].abs().mean().item()
                    variance_avg += state['exp_avg_sq'].mean().item()
                    count += 1
        if count > 0:
            momentum_avg /= count
            variance_avg /= count
        return momentum_avg, variance_avg

    def plot_training_dashboard(steps, losses, grad_norms, weight_norms,
                                 adam_momentums, adam_variances, perplexities):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Dashboard — Step {steps[-1]}', fontsize=16, fontweight='bold')

        # 1. Loss curve
        ax = axes[0, 0]
        ax.plot(steps, losses, color='#2196F3', linewidth=1.2)
        ax.set_title('Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Cross-Entropy Loss')
        ax.grid(True, alpha=0.3)

        # 2. Perplexity
        ax = axes[0, 1]
        ax.plot(steps, perplexities, color='#FF5722', linewidth=1.2)
        ax.set_title('Perplexity (e^loss)')
        ax.set_xlabel('Step')
        ax.set_ylabel('Perplexity')
        ax.grid(True, alpha=0.3)

        # 3. Gradient norm
        ax = axes[0, 2]
        ax.plot(steps, grad_norms, color='#4CAF50', linewidth=1.2)
        ax.axhline(y=gradient_clipping, color='red', linestyle='--', alpha=0.5, label=f'Clip={gradient_clipping}')
        ax.set_title('Gradient Norm')
        ax.set_xlabel('Step')
        ax.set_ylabel('L2 Norm')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Weight norm
        ax = axes[1, 0]
        ax.plot(steps, weight_norms, color='#9C27B0', linewidth=1.2)
        ax.set_title('Total Weight Norm')
        ax.set_xlabel('Step')
        ax.set_ylabel('L2 Norm')
        ax.grid(True, alpha=0.3)

        # 5. Adam momentum (exp_avg)
        ax = axes[1, 1]
        ax.plot(steps, adam_momentums, color='#FF9800', linewidth=1.2)
        ax.set_title('AdamW Momentum (avg |exp_avg|)')
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean Abs Value')
        ax.grid(True, alpha=0.3)

        # 6. Adam variance (exp_avg_sq)
        ax = axes[1, 2]
        ax.plot(steps, adam_variances, color='#607D8B', linewidth=1.2)
        ax.set_title('AdamW Variance (avg exp_avg_sq)')
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean Value')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_dashboard.png', dpi=100, bbox_inches='tight')
        plt.show()

    model.train()

    num_batches = 0
    while True:
        for batch in data.batch_items(data_iter, batch_size):
            if num_batches >= num_batches_to_train:
                break

            batch = batch.to(device)

            batch_x, batch_y = lm.batch_to_labeled_samples(batch)

            logits = model(batch_x)

            loss = lm.compute_loss(logits, batch_y)

            # parameters update
            model.zero_grad()
            loss.backward()

            # Record gradient norm BEFORE clipping
            grad_norm = compute_grad_norm(model)

            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            num_batches += 1

            # Log metrics every 10 steps
            if num_batches % 10 == 0:
                loss_val = loss.item()
                mom, var = compute_adam_stats(optimizer)
                steps_log.append(num_batches)
                loss_log.append(loss_val)
                grad_norm_log.append(grad_norm)
                weight_norm_log.append(compute_weight_norm(model))
                adam_momentum_log.append(mom)
                adam_variance_log.append(var)
                perplexity_log.append(math.exp(loss_val))

                print(f"Seen {num_batches} batches. loss: {loss_val:.4f}  ppl: {math.exp(loss_val):.2f}  grad_norm: {grad_norm:.4f}")

                if num_batches % 100 == 0:
                    model.eval()
                    sampled = tokenizer.detokenize(
                        model.better_sample_continuation(tokenizer.tokenize("Hello"), 500, temperature=0.8, topK=5)
                    )
                    model.train()
                    print(f"Model sample: '''{sampled}'''")
                    print("")

            # Plot dashboard periodically
            if num_batches % plot_every == 0 and len(steps_log) > 1:
                plot_training_dashboard(steps_log, loss_log, grad_norm_log,
                                        weight_norm_log, adam_momentum_log,
                                        adam_variance_log, perplexity_log)

    # Final dashboard
    if len(steps_log) > 1:
        print("\n=== Final Training Dashboard ===")
        plot_training_dashboard(steps_log, loss_log, grad_norm_log,
                                weight_norm_log, adam_momentum_log,
                                adam_variance_log, perplexity_log)
