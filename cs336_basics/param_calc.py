

# constants
d_model = 1600
d_vocab = 50257
num_layers = 48
d_ff = 6400

# Each transformer layer has:
# - 2 layer norms: 2 * d_model
# - Attention (Q,K,V,O): 4 * d_model^2  
# - SwiGLU FFN (gate, up, down): 3 * d_ff * d_model
params_tx_layers = num_layers*(2*d_model+4*d_model**2+
                               3*d_ff*d_model)

# Total: token embeddings (×2 for input and output) + final norm + transformer layers
total = 2*(d_model*d_vocab)+d_model+params_tx_layers

print(f"Number of params in Tx layers: {params_tx_layers}")
print(f"Total number parameters in Model: {total}")

# calculate the memory requirement:
total_mem = total*4 # 4 bytes for every float32 var
print(f"Total memory required {total_mem/1e9} GB")


# calculate memory required by AdamW
# Peak Memory = 4 × Parameters + Activations

# -----------------------------------------------
# Additional calculations: AdamW peak memory and activations (per spec)
# -----------------------------------------------

# Training setup / architecture knobs
num_heads = 25       # must divide d_model
context_length = 1024
batch_size = 4

DTYPE_BYTES = 4      # float32

# Parameter, gradient, and optimizer-state memory
params_bytes = total * DTYPE_BYTES
grads_bytes = total * DTYPE_BYTES
opt_state_bytes = 2 * total * DTYPE_BYTES  # AdamW m and v

# Activations (simplified, per block, per spec)
B = batch_size
S = context_length
H = num_heads

assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

# Per block: 16 * B * S * d_model token-shaped + 2 * (B * H * S * S) attention matrices
act_elems_per_block = B * S * (16 * d_model) + 2 * (B * H * S * S)
act_bytes_per_block = act_elems_per_block * DTYPE_BYTES

# Total activations (no activation checkpointing)
activations_bytes = num_layers * act_bytes_per_block

# Peak training memory (AdamW)
peak_bytes = params_bytes + grads_bytes + opt_state_bytes + activations_bytes

def to_gb(x: int) -> float:
    return x / 1e9

print("\n=== AdamW Memory Breakdown (float32) ===")
print(f"Parameters: {to_gb(params_bytes):.3f} GB")
print(f"Gradients:  {to_gb(grads_bytes):.3f} GB")
print(f"Opt state (AdamW m,v): {to_gb(opt_state_bytes):.3f} GB")
print(f"Per-block activations: {to_gb(act_bytes_per_block):.3f} GB")
print(f"Total activations ({num_layers} layers): {to_gb(activations_bytes):.3f} GB")
print(f"Peak = 4 * Parameters + Activations = {to_gb(peak_bytes):.3f} GB")
