#!/usr/bin/env python3
import torch
import sys
from pathlib import Path

def convert_model_to_cpu(input_path, output_path=None):
    """
    Loads a TorchScript model (possibly saved on CUDA) and re-saves it as a CPU model.
    Usage:
        python convert_model_to_cpu.py model_gpu.pt [model_cpu.pt]
    """
    input_path = Path(input_path)
    if output_path is None:
        # Default output name
        output_path = input_path.with_name(input_path.stem + "_cpu.pt")
    else:
        output_path = Path(output_path)
        
    model = torch.jit.load(str(input_path), map_location="cpu")
    model.to("cpu")
    
    # Optional sanity check
    for name, param in model.named_parameters():
        if param.device.type != "cpu":
            print(f"Warning: parameter {name} is still on {param.device}")
            break
        
    print(f"Saving CPU version to: {output_path}")
    torch.jit.save(model, str(output_path))
    
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_model_to_cpu.py <input_model.pt> [output_model.pt]")
        sys.exit(1)
                
    convert_model_to_cpu(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
