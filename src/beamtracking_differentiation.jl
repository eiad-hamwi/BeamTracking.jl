using ADTypes: AbstractADType, ForwardMode

"""
    AutoBeamTracking()

[`ADTypes.AbstractADType`](@ref) / DifferentiationInterface backend: Jacobian operators call BeamTracking's coordinate pushforward (analytic Jacobian), not AD through the primal map.

Use together with [`BeamTrackingFlow`](@ref).

## Second-order and higher (derivatives of the Jacobian)

Differentiate only the Jacobian by applying an outer backend to the Jacobian map, for example:

```julia
using DifferentiationInterface, ADTypes
using BeamTracking: beamtracking_flow, AutoBeamTracking

flow = beamtracking_flow(kernel; species=s, p_over_q_ref=pq)
jacflat(x) = vec(jacobian(flow, AutoBeamTracking(), x))  # length 36
H = jacobian(jacflat, AutoForwardDiff(), x)  # 36×6; no AD on `flow(x)`
```

With [GTPSA](https://github.com/bmadigan/GTPSA.jl) via `AutoGTPSA`, use a `GTPSA.Descriptor` whose `max_order` is large enough for the total differentiation order you need (e.g. order `3` recovers the same second-coordinate derivatives as an outer `AutoForwardDiff` on `jacflat`, and larger orders expose higher Taylor terms in one propagation). Example:

```julia
using GTPSA
d = Descriptor(6, 3)  # 6 state vars; adjust order for the target derivative degree
H = jacobian(jacflat, AutoGTPSA(; descriptor=d), x)
```

Or use [`beamtracking_jacobian_flat`](@ref) after loading DifferentiationInterface (weak extension).
"""
struct AutoBeamTracking <: AbstractADType end

ADTypes.mode(::AutoBeamTracking) = ForwardMode()

"""
    BeamTrackingFlow

Callable wrapper holding a kernel call or chain plus beam parameters. Use with DifferentiationInterface and [`AutoBeamTracking`](@ref).

# Fields
- `kcall_or_chain`: `KernelCall` or `KernelChain`
- `species`, `p_over_q_ref`, `t_ref`: passed to [`value_and_jacobian`](@ref) / tracking
- `track_kw`: `NamedTuple` of extra keyword arguments to [`track!`](@ref) on a `Bunch` with `jacobian=true` (e.g. `use_KA`, `use_explicit_SIMD`)

# Example

```julia
flow = beamtracking_flow(k; species=Species(\"electron\"), p_over_q_ref=-6e7)
y = flow(x)
J = jacobian(flow, AutoBeamTracking(), x)
```
"""
struct BeamTrackingFlow{K,Kw<:NamedTuple}
  kcall_or_chain::K
  species::Species
  p_over_q_ref::Float64
  t_ref::Float64
  track_kw::Kw
end

"""
    beamtracking_flow(kcall_or_chain; species=Species(), p_over_q_ref=NaN, t_ref=0.0, kwargs...)

Build a [`BeamTrackingFlow`](@ref). Any extra keyword arguments are forwarded to [`track!`](@ref) (`Bunch` + kernel chain) when evaluating the flow with [`AutoBeamTracking`](@ref).
"""
function beamtracking_flow(kcall_or_chain; species::Species=Species(), p_over_q_ref::Real=NaN, t_ref::Real=0.0, kwargs...)
  return BeamTrackingFlow(kcall_or_chain, species, float(p_over_q_ref), float(t_ref), NamedTuple(kwargs))
end

function (flow::BeamTrackingFlow)(x::AbstractVector)
  length(x) == 6 || error("BeamTrackingFlow expects a 6-element coordinate vector")
  return value_and_jacobian(
    x, flow.kcall_or_chain;
    species=flow.species, p_over_q_ref=flow.p_over_q_ref, t_ref=flow.t_ref, flow.track_kw...
  ).value
end

"""
    beamtracking_jacobian_flat(flow::BeamTrackingFlow)

After loading DifferentiationInterface, returns `x -> vec(jacobian(flow, AutoBeamTracking(), x))` for composing with outer AD (second derivatives of the Jacobian only).

If DifferentiationInterface is not loaded, this function has no methods.
"""
function beamtracking_jacobian_flat end
