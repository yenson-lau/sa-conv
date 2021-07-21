@Base.kwdef mutable struct GDOptimizer
  # Required
  θ
  grad_θx::Function

  # Hyperparameters
  η::Number = 0.1
  α::Union{Number, Nothing} = 0.9

  # Momentum variable
  θ₋ = nothing
end

function step!(opt::GDOptimizer; grad_kwargs...)::Nothing
  ω = if isa(opt.α, Nothing) || isa(opt.θ₋, Nothing)
    opt.θ
  else
    opt.θ + opt.α * (opt.θ - opt.θ₋)
  end

  grad = if length(grad_kwargs) > 0
    opt.grad_θx(ω; grad_kwargs)
  else
    opt.grad_θx(ω)
  end

  opt.θ₋ = opt.θ
  opt.θ = ω - opt.η * grad
  return
end
