import torch
from model.config import TSConfig
from model.modules.transformer import Transformer
import numpy as np
from utils import *


data_dir='data'
device='mps'
device_type='mps'

batch_size=4 ## Set the batch size according to GPU availability
config=TSConfig()
learning_rate = 1e-2
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
total_positions = len(train_data) - config.seq_len
total_batches = total_positions // batch_size


decoder_model=Transformer(config)
dm=decoder_model.to(device)
decoder_model=torch.compile(decoder_model,backend="eager")

optimizer = torch.optim.AdamW(decoder_model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_batches, eta_min=1e-7)

# Define checkpoint paths and frequency
checkpoint_dir = "checkpoints"
checkpoint_frequency = 80000000 # Set this according to the number of batches
os.makedirs(checkpoint_dir, exist_ok=True)

# Try to load the latest checkpoint if exists
latest_checkpoint = os.path.join(checkpoint_dir, "latest_checkpoint.pt")

start_iter = 0
if os.path.exists(latest_checkpoint):
    try:
        start_iter = load_checkpoint(latest_checkpoint, decoder_model, optimizer)
        # Reset scheduler to the right step
        for _ in range(start_iter):
            optimizer.step()
            scheduler.step()
        print(f"Resuming training from iteration {start_iter}")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")

total_iter = start_iter
decoder_model.train()
for batch_idx in range(start_iter, total_batches):
    total_iter += 1
    # Get sequential batch
    xb, yb = get_batch('train', batch_size, config.seq_len,device, batch_idx)
    # evaluate the loss
    logits, loss = decoder_model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(decoder_model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
    # Print loss after every step
    print(f"batch {batch_idx}/{total_batches}: loss {loss.item():.4f}, lr: {scheduler.get_last_lr()[0]:.6f}")
    
    #Enable this if you want checkpointing
    # Save checkpoint at regular intervals
    # if batch_idx % checkpoint_frequency == 0 or batch_idx == total_batches - 1:
    #     checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{batch_idx}.pt")
    #     save_checkpoint(decoder_model, optimizer, total_iter, checkpoint_path)
    #     # Also save as latest checkpoint
    #     save_checkpoint(decoder_model, optimizer, total_iter, latest_checkpoint)
    #     print(f"Saved checkpoint at iteration {total_iter}")
    if batch_idx ==100:
        break




