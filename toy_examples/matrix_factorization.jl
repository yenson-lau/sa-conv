using CUDA
using Flux
using LinearAlgebra
using Printf
using Random, Statistics
using TickTock

Random.seed!(1234)

m = 1000
X₀ = 5*ones(m, m)
X₁ = ones(m, m)
X = cat(X₀, X₁, dims=(1,2))

M₀ = cat(zeros(size(X₀)), X₁, dims=(1,2))
M₁ = cat(X₀, zeros(size(X₁)), dims=(1,2))

k = 2
U, V = randn(size(X,1), k), randn(size(X,2), k)
X̂ = U*V'

X, M₀, M₁, U, V, X̂ = cu(X), cu(M₀), cu(M₁), cu(U), cu(V), cu(X̂)

ℒ(U, V) = 1/2 * mean((U*V' .- X).^2)
ℒ(X̂) = 1/2 * mean((X̂ .- X).^2)
ℒ₀(X̂) = 1/2 * mean(abs.(M₀.*X̂))
ℒ₁(X̂) = 1/2 * mean(abs.(M₁.*X̂))

η, α = 1, 0.9
maxit, verbosity = 5000, 100
tick();
U₋, V₋ = U, V
for i ∈ 1:maxit
  # Apply acceleration -- convergence on bilinear optimization problems is extremely slow without it
  U₊ = U + α*(U - U₋)
  V₊ = V + α*(V - V₋)
  U₋, V₋ = U, V

  # Take gradient step
  ∇ᵤℒ, ∇ᵥℒ = gradient(ℒ, U₊, V₊)
  U = U₊ - η * ∇ᵤℒ
  V = V₊ - η * ∇ᵥℒ
  X̂ = U*V'

  if i % verbosity == 0
    @printf("(iter %6d, %.2fs)  ℒ: %.2e,  ℒ₀: %.2e,  ℒ₁: %.2e.\n",
            i, peektimer(), ℒ(X̂), ℒ₀(X̂), ℒ₁(X̂))
  end
end
tock();
