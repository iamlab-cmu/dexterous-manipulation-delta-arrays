import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.nn.functional as F
from einops import rearrange, reduce

# Assuming gpt_adaln_core.py, pytorch_warmup are in the same directory or installed
from gpt_adaln_core import Transformer
import pytorch_warmup as warmup

np.set_printoptions(precision=4)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# --- DATASET AND DATALOADER DEFINITIONS (Unchanged) ---

class ImitationDataset(Dataset):
    def __init__(self, states, actions, next_states, pos, num_agents, rewards, done):
        self.states = states
        self.actions = actions
        self.next_states = next_states
        self.pos = pos
        self.num_agents = num_agents
        self.rewards = rewards
        self.done = done

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        next_states = self.next_states[idx]
        pos = self.pos[idx]
        num_agents = self.num_agents[idx]
        reward = self.rewards[idx]
        done = self.done[idx]
        return state, action, next_states, pos, reward, done, num_agents

def get_smol_dataset(states, actions, next_states, pos, num_agents, rewards, done, num_samples:int=None):
    if num_samples is None:
        return ImitationDataset(states, actions, next_states, pos, num_agents, rewards, done)
        
    # Ensure we don't request more samples than available
    num_samples = min(num_samples, len(states))
    chosen_indices = np.random.choice(np.arange(len(states)), num_samples, replace=False)
    
    smol_states = states[chosen_indices]
    smol_actions = actions[chosen_indices]
    smol_next_states = next_states[chosen_indices]
    smol_pos = pos[chosen_indices]
    smol_num_agents = num_agents[chosen_indices]
    smol_rewards = rewards[chosen_indices]
    smol_done = done[chosen_indices]
    return ImitationDataset(smol_states, smol_actions, smol_next_states, smol_pos, smol_num_agents, smol_rewards, smol_done)

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset

def get_dataloaders_for_exp(
    actor_num_samples: int,
    rb_path: str,
    train_bs: int = 128,
    val_bs: int = 128,
    test_bs: int = 128
):
    """
    Loads data and creates dataloaders for a single experiment run.
    """
    with open(rb_path, 'rb') as f:
        replay_buffer = pkl.load(f)

    obs = replay_buffer['obs']
    act = replay_buffer['act']
    obs2 = replay_buffer['obs2']
    pos = replay_buffer['pos']
    num_agents = replay_buffer['num_agents']
    rewards = replay_buffer['rew']
    done = replay_buffer['done']

    # Actor: only actions with rewards > 30
    actor_idxs = np.where(rewards > 30)[0]
    actor_dataset = get_smol_dataset(
        obs[actor_idxs], act[actor_idxs], obs2[actor_idxs],
        pos[actor_idxs], num_agents[actor_idxs],
        rewards[actor_idxs], done[actor_idxs],
        num_samples=actor_num_samples
    )

    # Split actor dataset
    actor_train_dataset, actor_val_dataset, actor_test_dataset = split_dataset(actor_dataset)
    actor_train_loader = DataLoader(actor_train_dataset, batch_size=train_bs, shuffle=True)
    actor_val_loader = DataLoader(actor_val_dataset, batch_size=val_bs, shuffle=False)
    actor_test_loader = DataLoader(actor_test_dataset, batch_size=test_bs, shuffle=False)

    return actor_train_loader, actor_val_loader, actor_test_loader

# --- CORE TRAINING AND EVALUATION LOGIC ---

def evaluate_on_test_set(model, test_loader, hp_dict):
    """
    Computes the loss on the test dataloader.
    """
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for s1, a, _, p, _, _, N in test_loader:
            N = int(torch.max(N))
            
            actions = a[:, :N].to(hp_dict['dev_rl'])
            bs = actions.shape[0]
            # Match the action dimension for the model
            ones_column = -0.02 * torch.rand(bs, N, 1, device=hp_dict['dev_rl']) + 0.01
            actions_3d = torch.cat([actions, ones_column], dim=2)
            
            states = s1[:, :N].to(hp_dict['dev_rl'])
            pos = p[:, :N].to(hp_dict['dev_rl'])
            
            # Use the same loss function as in training for a fair comparison
            pi_loss = model.compute_actor_loss(actions_3d, states, pos)
            total_test_loss += pi_loss.item()
            
    return total_test_loss / len(test_loader)

def run_experiment(hp_dict, num_samples, n_epochs, test_interval, rb_path):
    """
    Runs a single training and evaluation experiment.
    
    Args:
        hp_dict (dict): Dictionary of hyperparameters.
        num_samples (int): Number of samples to use for training.
        n_epochs (int): Number of epochs to train for.
        test_interval (int): How often (in epochs) to evaluate on the test set.
        rb_path (str): Path to the replay buffer data.

    Returns:
        float: The final actor loss on the test set.
    """
    print(f"\n--- Running Experiment ---")
    print(f"Positional Embedding: {hp_dict['pos_embed']}, Num Samples: {num_samples}, Epochs: {n_epochs}")

    # 1. Get Dataloaders
    train_loader, _, test_loader = get_dataloaders_for_exp(
        actor_num_samples=num_samples,
        rb_path=rb_path,
        train_bs=256,
        test_bs=256
    )

    # 2. Initialize Model and Optimizer
    model = Transformer(hp_dict).to(hp_dict['dev_rl'])
    optimizer_actor = optim.AdamW(model.decoder_actor.parameters(), lr=hp_dict['pi_lr'], weight_decay=1e-2)
    lr_scheduler_actor = CosineAnnealingWarmRestarts(optimizer_actor, T_0=20, T_mult=2, eta_min=hp_dict['pi_eta_min'])

    # 3. Training Loop
    final_test_loss = []
    for epoch in range(n_epochs):
        model.train()
        pi_losses = []
        
        # Using tqdm for progress bar
        pbar = tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=False)
        for s1, a, s2, p, r, d, N in pbar:
            N = int(torch.max(N))
            actions = a[:, :N].to(hp_dict['dev_rl'])
            bs = actions.shape[0]
            ones_column = -0.02 * torch.rand(bs, N, 1, device=hp_dict['dev_rl']) + 0.01
            actions_3d = torch.cat([actions, ones_column], dim=2)
            states = s1[:, :N].to(hp_dict['dev_rl'])
            pos = p[:, :N].to(hp_dict['dev_rl'])
            
            optimizer_actor.zero_grad()
            pi_loss = model.compute_actor_loss(actions_3d, states, pos)
            pi_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer_actor.step()
            lr_scheduler_actor.step()
            
            pi_losses.append(pi_loss.item())
            pbar.set_postfix({"train_pi_loss": np.mean(pi_losses)}, refresh=True)

        # 4. Periodic Evaluation on Test Set
        if (epoch + 1) % test_interval == 0 or (epoch + 1) == n_epochs:
            test_loss = evaluate_on_test_set(model, test_loader, hp_dict)
            print(f"Epoch {epoch+1}: Test Actor Loss = {test_loss:.6f}")
            # The final loss is the one from the last epoch
            final_test_loss.append(test_loss)
            
    return final_test_loss

# --- PLOTTING ---

def plot_results(results, data_sizes, epoch_counts):
    """
    Generates and saves a 2x2 subplot of heatmaps, one for each position 
    embedding type, with a standardized and shared colorbar.
    """
    # --- Step 1: Find the global min and max loss for a standardized color scale ---
    all_losses = []
    for embed_type in results.keys():
        all_losses.extend(results[embed_type].values())
    
    if not all_losses:
        print("No results to plot.")
        return
        
    global_vmin = 0 
    global_vmax = 0.0010
    print(f"Standardizing color range from {global_vmin:.4f} to {global_vmax:.4f}")

    # --- Step 2: Create a 2x2 subplot grid ---
    # We create a figure and a 2x2 grid of axes
    fig, axs = plt.subplots(2, 2, figsize=(18, 15), sharex=True, sharey=True)
    # Flatten the 2x2 array of axes to make it easier to iterate through
    axs = axs.flatten()
    
    # --- Step 3: Iterate through results and plot on each subplot ---
    # We will store the last created image mappable to use for the colorbar
    mappable = None 
    pos_embed_types = list(results.keys())

    for i in range(4):
        ax = axs[i]
        
        # Make sure we don't try to plot more subplots than we have embedding types
        if i >= len(pos_embed_types):
            ax.set_visible(False) # Hide unused subplots
            continue

        embed_type = pos_embed_types[i]
        data = results[embed_type]
        
        # Create a grid to hold the loss values for the current embedding type
        loss_grid = np.full((len(epoch_counts), len(data_sizes)), np.nan)
        
        # Populate the grid
        for r, epochs in enumerate(epoch_counts):
            for c, samples in enumerate(data_sizes):
                if (samples, epochs) in data:
                    loss_grid[r, c] = data[(samples, epochs)]

        # Plotting the heatmap on the specific subplot (ax)
        # We use the 'coolwarm' colormap and the global vmin/vmax
        im = ax.imshow(loss_grid, cmap='coolwarm', vmin=global_vmin, vmax=global_vmax, interpolation='nearest', aspect='auto')
        mappable = im # Save the mappable object from the last plot

        # --- Step 4: Formatting for each subplot ---
        ax.set_title(f'Test Loss for {embed_type} Embedding', fontsize=14, weight='bold')
        
        # Set ticks (they are shared, so we only need to set labels once on the outer plots)
        ax.set_xticks(np.arange(len(data_sizes)))
        ax.set_yticks(np.arange(len(epoch_counts)))
        ax.set_xticklabels(data_sizes)
        ax.set_yticklabels(epoch_counts)
        
        # Add loss values to the cells
        for r in range(len(epoch_counts)):
            for c in range(len(data_sizes)):
                if not np.isnan(loss_grid[r, c]):
                    loss_val = loss_grid[r, c]
                    # Change text color for better readability based on background
                    text_color = 'white' if (loss_val - global_vmin) / (global_vmax - global_vmin) > 0.7 or (loss_val - global_vmin) / (global_vmax - global_vmin) < 0.3 else 'black'
                    ax.text(c, r, f'{loss_val*10000:.1f}', ha='center', va='center', color=text_color, fontsize=10)

    # --- Step 5: Add a single, shared colorbar and global labels ---
    fig.suptitle('Comparison of Positional Embeddings: Final Test Actor Loss', fontsize=20, weight='bold')
    
    # Add a single colorbar for the entire figure, linked to our mappable object
    # The colorbar will be placed to the right of the subplots
    fig.subplots_adjust(left=0.1, right=0.95, top=0.92, bottom=0.1, wspace=0.15, hspace=0.2)

    cbar_ax = fig.add_axes([0.9, 0.1, 0.01, 0.7])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label('Final Test Actor Loss', fontsize=12, weight='bold')
    
    # Add shared x and y labels for the entire figure
    fig.text(0.5, 0.05, 'Amount of Training Data (Number of Samples)', ha='center', va='center', fontsize=16)
    fig.text(0.07, 0.5, 'Computational Effort (Number of Epochs)', ha='center', va='center', rotation='vertical', fontsize=16)

    plt.tight_layout(rect=[0.08, 0.08, 0.9, 0.95]) # Adjust layout to make space for titles
    plt.savefig("all_embeddings_loss_comparison.png", dpi=300)
    print("Saved combined comparison heatmap to all_embeddings_loss_comparison.png")
    plt.show()
# --- MAIN EXECUTION BLOCK ---

if __name__ == "__main__":
    # A. Define Experiment Configurations
    EXPERIMENT_CONFIG = {
        "pos_embed_types": ['SCE', 'RoPE', 'Learned', 'SPE'],
        "data_sizes": [5, 25, 50, 75, 100, 150, 200, 350, 500, 
                       750, 1000, 1500, 2000, 3500, 5000, 7500, 10000], # Data required axis
        # "epoch_counts": [50, 100, 200, 400, 700, 1000],    # Compute effort axis
        "n_epochs": 500,
        "test_interval": 25, # How often to evaluate on test set
        "rb_path": '../../data/replay_buffer_mixed_obj.pkl'
    }

    # Base Hyperparameters (kept constant across experiments)
    base_hp_dict = {
        "exp_name": "MABC_Ablation",
        "pi_lr": 1e-4,
        "pi_eta_min": 1e-6,
        "dev_rl": device,
        "state_dim": 6,
        "action_dim": 3,
        'act_limit'         : 0.03,
        "model_dim": 256,
        "num_heads": 8,
        "dim_ff": 512,
        "n_layers_dict": {'encoder': 5, 'actor': 10, 'critic': 10},
        "dropout": 0,
        "max_grad_norm": 2.0,
        'idx_embed_loc': './idx_embedding_new.pth',
        "attn_mech": 'AdaLN',
        'gauss'             : True,
        'masked'            : True,
    }

    # Dictionary to store results: { 'SCE': {(samples, epochs): loss, ...}, ... }
    all_results = {embed_type: {} for embed_type in EXPERIMENT_CONFIG["pos_embed_types"]}

    # B. Run All Experiments
    for embed_type in EXPERIMENT_CONFIG["pos_embed_types"]:
        for num_samples in EXPERIMENT_CONFIG["data_sizes"]:
            # for n_epochs in EXPERIMENT_CONFIG["epoch_counts"]:
            # Create a specific config for this run
            if embed_type in ['SPE', 'RoPE']:
                base_hp_dict['attn_mech'] = "CA"
            current_hp = base_hp_dict.copy()
            current_hp['pos_embed'] = embed_type
            
            # Run the experiment
            final_losses = run_experiment(
                hp_dict=current_hp,
                num_samples=num_samples,
                n_epochs=EXPERIMENT_CONFIG["n_epochs"],
                test_interval=EXPERIMENT_CONFIG["test_interval"],
                rb_path=EXPERIMENT_CONFIG["rb_path"]
            )
            
            # Store the result
            for n_epochs, final_loss in enumerate(final_losses, start=1):
                all_results[embed_type][(num_samples, 
                            n_epochs*EXPERIMENT_CONFIG["test_interval"])] = final_loss

    # C. Generate Final Figures
    print("\n--- All experiments complete. Generating plots. ---")
    pkl.dump(all_results, open("mabc_test_gamut.pkl", "wb"))
    # all_results = pkl.load(open("mabc_test_gamut.pkl", "rb"))
    plot_results(
        results=all_results,
        data_sizes=EXPERIMENT_CONFIG["data_sizes"],
        epoch_counts=np.arange(0, 500, 25)
    )