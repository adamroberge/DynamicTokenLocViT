import torch


# Simulating the 'pack' function
def pack(tensors, format_str):
    packed_tensor = torch.cat(tensors, dim=1)
    # Record lengths of each tensor along the concatenation dimension
    positions = [t.shape[1] for t in tensors]
    return packed_tensor, positions


# Example tensors
batch_size = 32
num_patches = 196
num_register_tokens = 4
dim = 128

x = torch.randn(batch_size, num_patches, dim)
r = torch.randn(batch_size, num_register_tokens, dim)

# Packing tensors
packed_x, ps = pack([x, r], 'b * d')

# Output: torch.Size([32, 200, 128])
print("Packed tensor shape:", packed_x.shape)
print("Positions:", ps)  # Output: [196, 4]
