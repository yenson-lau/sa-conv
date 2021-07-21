@Base.kwdef mutable struct GDOptimizer
  # Required
  θ
  grad_θx::Function

  # Hyperparameters
  η::Number = 0.1
  α::Union{Number, Nothing} = 0.9
  num_iters::Integer = 100

  # Variables
  θ₋ = nothing
  iter = 0
end

function step!(opt::GDOptimizer, x)::Nothing
  ω = if isa(opt.α, Nothing) || isa(opt.θ₋, Nothing)
    opt.θ
  else
    opt.θ + opt.α * (opt.θ - opt.θ₋)
  end

  opt.θ₋ = opt.θ
  opt.θ = ω - opt.η * opt.grad_θx(ω, x)
  return
end