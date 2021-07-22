@Base.kwdef mutable struct SelfAttention
  # Hyperparameters
  Nₕ        # number of heads
  D_in      # input dimension
  D_out     # (final) output dimension
  D_key     # key dimension
  D_h       # head dimension
  P         # positional encoding

  # Learnable parameters
  W_query
  W_key
  W_val
  W_out
  b_out
end

function (sa::SelfAttention)(X)
  X₊P = X + sa.P
  for h ∈ 1:sa.Nₕ
    A[:,:,h] = X₊P * sa.W_query[:,:,h] * sa.W_key[:,:,h]' * X₊P'
  end

  self_attn(h) = 0    # TODO: finish this function
  concat_heads = cat((self_attn(h) for h ∈ 1:sa.Nₕ)...; dims=2)
  return concat_heads * sa.W_out + sa.b_out
end

# TODO: How to pass parameters / get gradient?
