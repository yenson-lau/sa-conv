using CUDA
using Flux
using LinearAlgebra
using Printf
using Random, Statistics
using TickTock

Random.seed!(1234)

m, n = 2000, 1000;
x₀, A = randn(n)/sqrt(n), randn(m, n)/sqrt(m);
y = A*x₀;

x̂ = randn(n)/sqrt(n);
# A, x̂, y = cu(A), cu(x̂), cu(y)  # CUDA

loss(x) = 1/2 * sum((A*x .- y).^2);
∇loss(x) = gradient(loss, x)[1]
η = 1/norm(A)

println(loss(x̂))
println(size(∇loss(x̂)))

tick();
for i ∈ 1:1000
  x̂ = x̂ - η * ∇loss(x̂);
  if i%100 == 0
    @printf("(iter %4d) loss: %.4e, elapsed: %.3fs\n",
            i, loss(x̂), peektimer());
  end
end
tock();
