import torch 
from model.modules.embedding import Embedding
import os
import numpy as np

data_dir='data'

def save_checkpoint(model, optimizer, iteration, out):
    """
    Save model, optimizer, and iteration state to a file.
    
    Args:
        model: torch.nn.Module - The model to save
        optimizer: torch.optim.Optimizer - The optimizer to save
        iteration: int - Current iteration number
        out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes] - Output file or path
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer):
    """
    Load model, optimizer state from a checkpoint file.
    
    Args:
        src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes] - Input file or path
        model: torch.nn.Module - The model to load state into
        optimizer: torch.optim.Optimizer - The optimizer to load state into
        
    Returns:
        int: The iteration number saved in the checkpoint
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']

def count_parameters(model, skip_embedding=True):
    """
    Count the total number of trainable parameters in a PyTorch model,
    excluding custom Embedding layers.
    
    Args:
        model: torch.nn.Module - The model to analyze
        skip_embedding: bool - Whether to exclude custom Embedding layers
        
    Returns:
        int: Total number of trainable parameters
        
    Example:
        model = Transformer(config)
        print(f"Non-embedding parameters: {count_parameters(model):,}")
        print(f"All parameters: {count_parameters(model, skip_embedding=False):,}")
    """
    if skip_embedding:
        return sum(p.numel() for name, p in model.named_parameters() 
                  if p.requires_grad and not any(isinstance(module, Embedding) 
                                              for name_prefix, module in model.named_modules() 
                                              if name.startswith(name_prefix)))
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_batch(split, batch_size, seq_len,device,batch_idx=None):
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    total_positions = len(data) - seq_len
    if batch_idx is None:
        ix = torch.randint(total_positions, (batch_size,))
    else:
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_positions)
        ix = torch.arange(start_idx, end_idx)
        if len(ix) < batch_size:
            padding = torch.zeros(batch_size - len(ix), dtype=torch.long)
            ix = torch.cat([ix, padding])
    x = torch.stack([torch.from_numpy((data[i:i+seq_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+seq_len]).astype(np.int64)) for i in ix])  
    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y