using Flux
using LinearAlgebra
using Random
using Printf
using TickTock
include("../utils/GradientDescent.jl");

## Generate data / define mappings
k, n, θ, N = 5, 50, 0.2, 100
Random.seed!(1234)
a₀ = collect(1:k) * 1.
X₀ = (rand(n, N) .<= θ) .* 1.
# X₀ = zeros(n, N);  X₀[1,:] .= 1;  X₀

function ⋆⃑(u::Vector, V::Matrix; pad::Integer=k-1)::Matrix
  return NNlib.conv(reshape(V, size(V,1), :, size(V,2)),
                    reshape(u, :,1,1);
                    pad=pad)[:,1,:]
end

# Matrix operations
function toeplitz(u::Vector, T_size::Integer)::Matrix
  @assert sz >= length(u) "T_size must be geq to length(u)"
  u_padded = [u; zeros(T_size - length(u) + T_size)]

  T = zeros(length(u_padded), T_size)
  for (col, shift) ∈ enumerate(0:T_size-1)
    T[:,col] = circshift(u_padded, shift)
  end
  return T[1:T_size, :]
end

function ⋆⃑(U::Matrix, V::Matrix; pad::Integer=k-1)::Matrix
  U_ = [[U zeros(size(U,1), pad)]; zeros(pad, size(U,2)+pad)]
  V_ = [V; zeros(pad, size(V,2))]
  return U_*V_
end

A₀ = toeplitz(a₀, n)
Y = a₀ ⋆⃑ X₀
# plot(X₀[:,1], line=:stem, marker=:dot, label="x₀", markersize=2)
# plot!(Y[:,1], line=:stem, marker=:dot, label="y", markersize=2)

ℒ(a, X) = sum((a ⋆⃑ X - Y).^2)/2
∇ℒ(a, X) = gradient(ℒ, a, X)
nz(A; thresh=1e-5) = A .* (abs.(A).>thresh)


## Optimize
ℒ̂(a::Vector) = norm(a-a₀)
ℒ̂(A::Matrix) = norm(A-A₀)

# A = randn(k)      # recover kernel vector a₀
A = randn(n, n)   # recover toeplitz matrix A₀
opt = GDOptimizer(
  grad_θx = A -> ∇ℒ(A, X₀)[1],
  θ = A,
  η = 0.001,
  α = 0.9
)

tick();
for i ∈ 1:1000
  global opt, A, A₀
  step!(opt)
  A = opt.θ

  if i ∈ [1:10;] .* 10^Integer(floor(log10(i)))
    @printf("(iter %4d) loss: (%.4e, %.4e), elapsed: %.3fs\n",
            i, ℒ(A, X₀), ℒ̂(A), peektimer());
  end
end
tock();
