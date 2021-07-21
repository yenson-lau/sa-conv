using Flux
using Plots

k, n, θ = 5, 50, 0.1

a₀ = collect(1:k) * 1.
x₀ = (rand(n) .<= θ) * 1.
x₀ = zeros(n);  x₀[1] = 1

conv(a, x) = NNlib.conv(
  reshape(x, :,1,1), reshape(a, :,1,1);
  pad=k-1
)[:,1,1]
y = conv(a₀, x₀)

# plot(x₀, line=:stem, marker=:dot, label="x₀")
# plot!(y, line=:stem, marker=:dot, label="y")

function toeplitz(a::Vector, sz::Integer)::Matrix
  @assert sz >= length(a) "matrix size sz must be geq to size(a)"
  a_padded = [a; zeros(sz - length(a) + sz)]

  T = zeros(length(a_padded), sz)
  for (col, shift) ∈ enumerate(0:sz-1)
    T[:,col] = circshift(a_padded, shift)
  end
  return T[1:sz, :]
end

ℒ(a::Vector, x::Vector) = sum((conv(a, x) - y).^2)/2
ℒ(A::Matrix, x::Vector) = sum((A*[x; zeros(k-1)] - y).^2)/2
∇ℒ(a, x) = gradient(ℒ, a, x)

A = toeplitz(a₀, n+k-1)
∇ℒ(A, x₀)
