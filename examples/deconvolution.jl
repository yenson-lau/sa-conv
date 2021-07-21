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
X₀ = (rand(n, N) .<= θ) .* 1
# X₀ = zeros(n, N);  X₀[1,:] .= 1;  X₀

function ⋆(a::Vector, X::Matrix)::Matrix
  return NNlib.conv(reshape(X, size(X,1), :, size(X,2)),
                    reshape(a, :,1,1);
                    pad=k-1)[:,1,:]
end

# Matrix operations
function toeplitz(a::Vector, sz::Integer)::Matrix
  @assert sz >= length(a) "matrix size sz must be geq to size(a)"
  a_padded = [a; zeros(sz - length(a) + sz)]

  T = zeros(length(a_padded), sz)
  for (col, shift) ∈ enumerate(0:sz-1)
    T[:,col] = circshift(a_padded, shift)
  end
  return T[1:sz, :]
end

function ⋆(A::Matrix, X::Matrix)::Matrix
  A_ = [[A zeros(size(A,1), k-1)]; zeros(k-1, size(A,2)+k-1)]
  X_ = [X; zeros(k-1, size(X,2))]
  return A_*X_
end

A₀ = toeplitz(a₀, n)
Y = a₀ ⋆ X₀
# plot(X₀[:,1], line=:stem, marker=:dot, label="x₀", markersize=2)
# plot!(Y[:,1], line=:stem, marker=:dot, label="y", markersize=2)

ℒ(a, X) = sum((a ⋆ X - Y).^2)/2
∇ℒ(a, X) = gradient(ℒ, a, X)
nz(A; thresh=1e-5) = A .* (abs.(A).>thresh)


## Optimize
ℒ̂(a::Vector) = norm(a-a₀)
ℒ̂(A::Matrix) = norm(A-A₀)

# A = randn(k)      # recover kernel vector a₀
A = randn(n, n)   # recover toeplitz matrix A₀
opt = GDOptimizer(
  grad_θx = (A,_) -> ∇ℒ(A, X₀)[1],
  θ = A,
  η = 0.001,
  α = 0.9
)

tick();
for i ∈ 1:1000
  global opt, A, A₀
  step!(opt, nothing)
  A = opt.θ

  if i ∈ [1:10;] .* 10^Integer(floor(log10(i)))
    @printf("(iter %4d) loss: (%.4e, %.4e), elapsed: %.3fs\n",
            i, ℒ(A, X₀), ℒ̂(A), peektimer());
  end
end
tock();
