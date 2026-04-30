"""
KernelChain host-side conversion and validation utilities.

These helpers are intentionally not called from `launch!` to keep the hot path
allocation-free and free of backend-specific policy.
"""

# --- Float leaf conversion (widening prior) ---

@inline convert_eltype(kcall::KernelCall, ::Type{T}) where {T<:AbstractFloat} =
  KernelCall(kcall.kernel, _convert_eltype_leaf(kcall.args, T))

@inline convert_eltype(kc::KernelChain, ::Type{T}) where {T<:AbstractFloat} =
  KernelChain(map(k -> convert_eltype(k, T), kc.chain), _convert_eltype_leaf(kc.ref, T))

@inline _convert_eltype_leaf(x::Nothing, ::Type{T}) where {T<:AbstractFloat} = x
@inline _convert_eltype_leaf(x::Bool, ::Type{T}) where {T<:AbstractFloat} = x
@inline _convert_eltype_leaf(x::Integer, ::Type{T}) where {T<:AbstractFloat} = x
@inline _convert_eltype_leaf(x::AbstractFloat, ::Type{T}) where {T<:AbstractFloat} = T(x)
@inline _convert_eltype_leaf(x::Tuple, ::Type{T}) where {T<:AbstractFloat} = map(y -> _convert_eltype_leaf(y, T), x)

@inline function _convert_eltype_leaf(x::StaticArray, ::Type{T}) where {T<:AbstractFloat}
  return eltype(x) <: AbstractFloat ? map(T, x) : x
end

@inline _convert_eltype_leaf(x::RefState, ::Type{T}) where {T<:AbstractFloat} =
  RefState(_convert_eltype_leaf(x.t, T), _convert_eltype_leaf(x.beta_gamma, T))

@inline _convert_eltype_leaf(x, ::Type{T}) where {T<:AbstractFloat} = x

# --- Backend compatibility validation (generic; host-side) ---

@inline function validate_kernelchain(
  coords::Coords,
  kc::KernelChain;
  groupsize::Union{Nothing,Integer}=nothing,
  use_KA::Bool=!(get_backend(coords.v) isa CPU && isnothing(groupsize)),
)
  backend = get_backend(coords.v)

  # Widening prior: convert only AbstractFloat leaves to eltype(coords.v)
  T = eltype(coords.v)
  kc2 = (use_KA && T <: AbstractFloat) ? convert_eltype(kc, T) : kc

  if use_KA && !(backend isa CPU)
    _validate_backend_compat(backend, coords, kc2)
  end
  return kc2
end

@inline _is_metal_backend(backend) = occursin("Metal", string(typeof(backend)))

function _validate_backend_compat(backend, coords::Coords, kc::KernelChain)
  if _is_metal_backend(backend) && eltype(coords.v) === Float64
    error("Metal backend does not support Float64 kernels; use Float32 coordinates (eltype(coords.v)=Float32).")
  end

  bad = _first_unsupported_gpu_number(kc)
  if bad !== nothing
    error("KernelChain contains a non-GPU-compatible numeric leaf at $bad. " *
          "On GPU backends, kernel argument trees may only contain AbstractFloat/Integer/Bool numeric leaves " *
          "(and tuples/static arrays/structs thereof).")
  end
  return nothing
end

@inline _gpu_number_supported(x::AbstractFloat) = true
@inline _gpu_number_supported(x::Integer) = true
@inline _gpu_number_supported(x::Bool) = true
@inline _gpu_number_supported(x::Number) = false

function _first_unsupported_gpu_number(x; path::String="root")
  if x === nothing
    return nothing
  elseif x isa Number
    return _gpu_number_supported(x) ? nothing : "$(path) :: $(typeof(x))"
  elseif x isa Tuple
    for (i, xi) in pairs(x)
      r = _first_unsupported_gpu_number(xi; path="$(path)[$i]")
      r === nothing || return r
    end
    return nothing
  elseif x isa NamedTuple
    for k in keys(x)
      r = _first_unsupported_gpu_number(getfield(x, k); path="$(path).$(k)")
      r === nothing || return r
    end
    return nothing
  elseif x isa StaticArray
    for i in eachindex(x)
      r = _first_unsupported_gpu_number(x[i]; path="$(path)[$i]")
      r === nothing || return r
    end
    return nothing
  elseif x isa RefState
    r = _first_unsupported_gpu_number(x.t; path="$(path).t")
    r === nothing || return r
    return _first_unsupported_gpu_number(x.beta_gamma; path="$(path).beta_gamma")
  else
    Tx = typeof(x)
    if Base.isstructtype(Tx) && Base.isbitstype(Tx) && fieldcount(Tx) > 0
      fns = fieldnames(Tx)
      for i in 1:fieldcount(Tx)
        fn = fns[i]
        r = _first_unsupported_gpu_number(getfield(x, i); path="$(path).$(fn)")
        r === nothing || return r
      end
    end
    return nothing
  end
end

@inline function _first_unsupported_gpu_number(kc::KernelChain)
  for (i, kcall) in pairs(kc.chain)
    r = _first_unsupported_gpu_number(kcall.args; path="kc.chain[$i].args")
    r === nothing || return r
  end
  return _first_unsupported_gpu_number(kc.ref; path="kc.ref")
end

