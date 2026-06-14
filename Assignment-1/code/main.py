from __future__ import annotations

import os
import math
import torch

if __name__ == "__main__":
    import lm
    import data
    import matplotlib.pyplot as plt
    from torch import optim
    from transformer import TransformerLM

    # -----------------------------
    # Training configuration
    # -----------------------------
    seq_len = 128
    batch_size = 64
    data_path = "/content/drive/MyDrive/LM Course/Assignment-1/data/he"

    n_layers = 6
    n_heads = 6
    embed_size = 192
    mlp_hidden_size = embed_size * 4

    learning_rate = 5e-4
    gradient_clipping = 1.0
    num_batches_to_train = 50000

    plot_every = 500
    save_metrics_every = 10

    target_loss = 0.25
    min_batches_before_stopping = 10000
    stop_when_target_reached = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # -----------------------------
    # Output folders
    # -----------------------------
    base_dir = "/content/drive/MyDrive/LM Course/Assignment-1"
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    report_dir = os.path.join(base_dir, "reports")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_model.pt")

    # -----------------------------
    # Data
    # -----------------------------
    tokenizer, tokenized_data = data.load_data(data_path)
    data_iter = iter(data.RandomOrderDataIterator(tokenized_data, seq_len + 1))

    # -----------------------------
    # Model
    # -----------------------------
    model: torch.nn.Module = TransformerLM(
        n_layers=n_layers,
        n_heads=n_heads,
        embed_size=embed_size,
        max_context_len=seq_len,
        vocab_size=tokenizer.vocab_size(),
        mlp_hidden_size=mlp_hidden_size,
        with_residuals=True,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=[0.9, 0.95])

    # -----------------------------
    # Metrics storage
    # -----------------------------
    steps_log = []
    loss_log = []
    grad_norm_log = []
    weight_norm_log = []
    adam_momentum_log = []
    adam_variance_log = []
    perplexity_log = []

    best_loss = float("inf")
    should_stop = False

    # -----------------------------
    # Helper functions
    # -----------------------------
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
        momentum_avg = 0.0
        variance_avg = 0.0
        count = 0

        for group in optimizer.param_groups:
            for p in group["params"]:
                state = optimizer.state.get(p)
                if state and "exp_avg" in state:
                    momentum_avg += state["exp_avg"].abs().mean().item()
                    variance_avg += state["exp_avg_sq"].mean().item()
                    count += 1

        if count > 0:
            momentum_avg /= count
            variance_avg /= count

        return momentum_avg, variance_avg

    def save_checkpoint(path, step, loss_value):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss_value,
                "step": step,
                "config": {
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "n_layers": n_layers,
                    "n_heads": n_heads,
                    "embed_size": embed_size,
                    "mlp_hidden_size": mlp_hidden_size,
                    "learning_rate": learning_rate,
                    "vocab_size": tokenizer.vocab_size(),
                    "with_residuals": True,
                },
            },
            path,
        )

    def save_metrics_csv():
        csv_path = os.path.join(report_dir, "training_metrics.csv")

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("step,loss,perplexity,grad_norm,weight_norm,adam_momentum,adam_variance\n")
            for i in range(len(steps_log)):
                f.write(
                    f"{steps_log[i]},"
                    f"{loss_log[i]},"
                    f"{perplexity_log[i]},"
                    f"{grad_norm_log[i]},"
                    f"{weight_norm_log[i]},"
                    f"{adam_momentum_log[i]},"
                    f"{adam_variance_log[i]}\n"
                )

    def plot_training_dashboard():
        if len(steps_log) < 2:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            f"Training Dashboard — Step {steps_log[-1]}",
            fontsize=16,
            fontweight="bold",
        )

        # Loss
        ax = axes[0, 0]
        ax.plot(steps_log, loss_log, linewidth=1.2)
        ax.set_title("Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.grid(True, alpha=0.3)

        # Perplexity
        ax = axes[0, 1]
        ax.plot(steps_log, perplexity_log, linewidth=1.2)
        ax.set_title("Perplexity")
        ax.set_xlabel("Step")
        ax.set_ylabel("e^loss")
        ax.grid(True, alpha=0.3)

        # Gradient norm
        ax = axes[0, 2]
        ax.plot(steps_log, grad_norm_log, linewidth=1.2)
        ax.axhline(
            y=gradient_clipping,
            linestyle="--",
            alpha=0.5,
            label=f"clip={gradient_clipping}",
        )
        ax.set_title("Gradient Norm")
        ax.set_xlabel("Step")
        ax.set_ylabel("L2 Norm")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Weight norm
        ax = axes[1, 0]
        ax.plot(steps_log, weight_norm_log, linewidth=1.2)
        ax.set_title("Total Weight Norm")
        ax.set_xlabel("Step")
        ax.set_ylabel("L2 Norm")
        ax.grid(True, alpha=0.3)

        # Adam momentum
        ax = axes[1, 1]
        ax.plot(steps_log, adam_momentum_log, linewidth=1.2)
        ax.set_title("AdamW Momentum avg |exp_avg|")
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Absolute Value")
        ax.grid(True, alpha=0.3)

        # Adam variance
        ax = axes[1, 2]
        ax.plot(steps_log, adam_variance_log, linewidth=1.2)
        ax.set_title("AdamW Variance avg exp_avg_sq")
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean Value")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        dashboard_path = os.path.join(
            report_dir,
            f"training_dashboard_step_{steps_log[-1]}.png",
        )
        plt.savefig(dashboard_path, dpi=120, bbox_inches="tight")
        plt.show()

    # -----------------------------
    # Training loop
    # -----------------------------
    model.train()
    num_batches = 0

    while True:
        for batch in data.batch_items(data_iter, batch_size):
            if num_batches >= num_batches_to_train:
                should_stop = True
                break

            batch = batch.to(device)
            batch_x, batch_y = lm.batch_to_labeled_samples(batch)

            logits = model(batch_x)
            loss = lm.compute_loss(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()

            grad_norm = compute_grad_norm(model)

            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            num_batches += 1

            # -----------------------------
            # Logging + checkpointing
            # -----------------------------
            if num_batches % save_metrics_every == 0:
                loss_val = loss.item()
                ppl = math.exp(loss_val)
                mom, var = compute_adam_stats(optimizer)

                steps_log.append(num_batches)
                loss_log.append(loss_val)
                perplexity_log.append(ppl)
                grad_norm_log.append(grad_norm)
                weight_norm_log.append(compute_weight_norm(model))
                adam_momentum_log.append(mom)
                adam_variance_log.append(var)

                print(
                    f"Seen {num_batches} batches. "
                    f"loss: {loss_val:.4f}  "
                    f"ppl: {ppl:.2f}  "
                    f"grad_norm: {grad_norm:.4f}"
                )

                if loss_val < best_loss:
                    best_loss = loss_val
                    save_checkpoint(best_checkpoint_path, num_batches, best_loss)
                    print(f"Saved best model: loss={best_loss:.4f}, step={num_batches}")

                if num_batches % 100 == 0:
                    model.eval()
                    sampled = tokenizer.detokenize(
                        model.better_sample_continuation(
                            tokenizer.tokenize("Hello"),
                            500,
                            temperature=0.8,
                            topK=5,
                        )
                    )
                    model.train()
                    print(f"Model sample: '''{sampled}'''")
                    print("")

                if (
                    stop_when_target_reached
                    and num_batches >= min_batches_before_stopping
                    and loss_val <= target_loss
                ):
                    print(f"Target loss reached: {loss_val:.4f} at step {num_batches}")
                    should_stop = True
                    break

            # -----------------------------
            # Periodic plots
            # -----------------------------
            if num_batches % plot_every == 0:
                save_metrics_csv()
                plot_training_dashboard()

        if should_stop:
            break

    # -----------------------------
    # Final save
    # -----------------------------
    final_loss = loss.item()
    save_checkpoint(final_checkpoint_path, num_batches, final_loss)
    save_metrics_csv()
    plot_training_dashboard()

    print("\nTraining finished.")
    print(f"Final step: {num_batches}")
    print(f"Final loss: {final_loss:.4f}")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Final checkpoint: {final_checkpoint_path}")
    print(f"Reports folder: {report_dir}")
