
const REGISTER_SIZE = register_size()

# This is here in case kernel chain needs to be run 
# but is not fully filled. It does nothing
blank_kernel!(args...) = nothing

@kwdef struct KernelCall{K,A}
  kernel::K = blank_kernel!
  args::A   = ()
  function KernelCall(kernel, args)
    _args = map(t->time_lower(batch_lower(t)), args)
    new{typeof(kernel),typeof(_args)}(kernel, _args)
  end 
end

# In case KernelCall contains batch GPU array
Adapt.@adapt_structure KernelCall

# Store the state of the reference coordinate system
# Needed for time-dependent parameters
struct RefState{T,U}
  t::T          # Reference time
  beta_gamma::U # Reference energy
end

# Alias
struct KernelChain{C<:Tuple{Vararg{<:KernelCall}}, S<:Union{Nothing,RefState}}
  chain::C  # The tuple of KernelCalls
  ref::S    # An optional RefState for the initial time-dependent parameters
  KernelChain(chain, ref=nothing) = new{typeof(chain), typeof(ref)}(chain, ref)
end

# In case KernelChain contains batch GPU array
Adapt.@adapt_structure KernelChain

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

# Opt-in host-side validator for KernelChains before launching on KA/GPU backends.
# This is intentionally NOT called from `launch!` to keep the hot path allocation-free.
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

KernelChain(::Val{N}, ref=nothing) where {N} = KernelChain(ntuple(t->KernelCall(), Val{N}()), ref)

push(kc::KernelChain, kcall::Nothing) = kc

push(kc::KernelChain, kcall) = @reset kc.chain = _push(kc.chain, kcall)

@unroll function _push(chain, kcall)
  i = 0
  @unroll for kcalli in chain
    i += 1
    if kcalli.kernel == blank_kernel!
      return @reset chain[i] = kcall
    end
  end
  error("Unable to push KernelCall to kernel chain: kernel chain is full")
end

# KA does not like Vararg
@kernel function generic_kernel!(coords::Coords, @Const(kc::KernelChain))
  i = @index(Global, Linear)
  @inline _generic_kernel!(i, coords, kc)
end

_generic_kernel!(i, coords, kc) = __generic_kernel!(i, coords, kc.chain, kc.ref)

@unroll function __generic_kernel!(i, coords::Coords, chain, ref)
  @unroll for kcall in chain
    bargs = process_batch_args(i, kcall.args)
    args = process_time_args(i, coords, bargs, ref)
    (kcall.kernel)(i, coords, args...)
  end
  return nothing
end

function process_batch_args(i, args)
  if static_batchcheck(args) 
    return beval(args, i)
  else
    return args
  end
end

function process_time_args(i, coords, args, ref)
  if !isnothing(ref) && static_timecheck(args) 
    let t = compute_time(coords.v[i,ZI], coords.v[i,PZI], ref)
      return teval(args, t)
    end
  else
    return args
  end
end

# Function to execute callbacks
execute_callbacks(coords, ds_step, g) = _execute_callbacks(coords.callbacks, coords, ds_step, g)

@unroll function _execute_callbacks(callbacks, coords, ds_step, g)
  @unroll for callback in callbacks
    callback(coords, ds_step, g)
  end
  return nothing
end

# Generic function to launch a kernel on the bunch coordinates matrix
# Matrix v should ALWAYS be in SoA whether for real or as a view via tranpose(v)
# Primal if `coords.jac === nothing`; else Jacobian pushforward (`jacobianize` + same `generic_kernel!` driver).

@inline function _launch_body!(
  coords::Coords{<:Any,V,<:Any,<:Any,<:Any,<:Any},
  kc::KernelChain;
  groupsize::Union{Nothing,Integer},
  multithread_threshold::Integer,
  use_KA::Bool,
  use_explicit_SIMD::Bool,
) where {V}
  v = coords.v
  N_particle = size(v, 1)

  if use_KA && use_explicit_SIMD
    error("Cannot use both KernelAbstractions (KA) and explicit SIMD")
  end

  if !use_KA
    if use_explicit_SIMD && V <: SIMD.FastContiguousArray && eltype(V) <: SIMD.ScalarTypes && pick_vector_width(eltype(V)) > 1
      simd_lane_width = pick_vector_width(eltype(V))
      lane = SIMD.VecRange{Int(simd_lane_width)}(0)
      rmn = rem(N_particle, simd_lane_width)
      N_SIMD = N_particle - rmn
      if N_particle >= multithread_threshold
        Threads.@threads for i in 1:simd_lane_width:N_SIMD
          @assert last(i) <= N_particle "Out of bounds!"
          _generic_kernel!(lane+i, coords, kc)
        end
      else
        for i in 1:simd_lane_width:N_SIMD
          @assert last(i) <= N_particle "Out of bounds!"
          _generic_kernel!(lane+i, coords, kc)
        end
      end
      for i in N_SIMD+1:N_particle
        @assert last(i) <= N_particle "Out of bounds!"
        _generic_kernel!(i, coords, kc)
      end
    else
      if N_particle >= multithread_threshold
        Threads.@threads for i in 1:N_particle
          @assert last(i) <= N_particle "Out of bounds!"
          _generic_kernel!(i, coords, kc)
        end
      else
        @simd for i in 1:N_particle
          @assert last(i) <= N_particle "Out of bounds!"
          _generic_kernel!(i, coords, kc)
        end
      end
    end
  else
    backend = get_backend(v)
    if isnothing(groupsize)
      kernel! = generic_kernel!(backend)
    else
      kernel! = generic_kernel!(backend, groupsize)
    end
    kernel!(coords, kc; ndrange=N_particle)
    KernelAbstractions.synchronize(backend)
  end
  return nothing
end

@inline function launch!(
  coords::Coords{<:Any,V,<:Any,<:Any,<:Any,Nothing},
  kc::KernelChain;
  groupsize::Union{Nothing,Integer}=nothing,
  multithread_threshold::Integer=Threads.nthreads() > 1 ? 1750*Threads.nthreads() : typemax(Int),
  use_KA::Bool=!(get_backend(coords.v) isa CPU && isnothing(groupsize)),
  use_explicit_SIMD::Bool=!use_KA,
) where {V}
  return _launch_body!(
    coords,
    kc;
    groupsize,
    multithread_threshold,
    use_KA,
    use_explicit_SIMD,
  )
end

@inline function launch!(
  coords::Coords{<:Any,V,<:Any,<:Any,<:Any,<:Any},
  kc::KernelChain;
  groupsize::Union{Nothing,Integer}=nothing,
  multithread_threshold::Integer=Threads.nthreads() > 1 ? 1750*Threads.nthreads() : typemax(Int),
  use_KA::Bool=!(get_backend(coords.v) isa CPU && isnothing(groupsize)),
  use_explicit_SIMD::Bool=!use_KA,
) where {V}
  kc_eff = jacobianize(kc)
  preflight_jacobian_tracking(coords, kc_eff; use_KA=use_KA, use_explicit_SIMD=use_explicit_SIMD)
  return _launch_body!(
    coords,
    kc_eff;
    groupsize,
    multithread_threshold,
    use_KA,
    use_explicit_SIMD,
  )
end

function check_kwargs(mac, kwargs...)
  valid_kwargs = [:(fastgtpsa)=>Bool, :(inbounds)=>Bool]
  for k in kwargs
    if Meta.isexpr(k, :(=))
      pk = Pair(k.args...)
      idx = findfirst(t->t==pk[1], map(t->t[1], valid_kwargs))
      if isnothing(idx)
        error("Unrecognized input to @$(mac) macro: $(pk[1])")
      elseif typeof(pk[2]) != valid_kwargs[idx][2]
        error("Type for keyword argument `$(pk[1])` must be `$(valid_kwargs[idx][2])`")
      end
    else
      error("Unrecognized input to @$(mac) macro: $k")
    end
  end
end

# Also allow launch! on single KernelCalls
@inline launch!(coords::Coords, kcall::KernelCall; kwargs...) = launch!(coords, KernelChain((kcall,)); kwargs...)

macro makekernel(args...)
  kwargs = args[1:length(args)-1]
  fcn = last(args)

  fcn.head == :function || error("@makekernel must wrap a function definition")
  body = esc(fcn.args[2])
  signature = fcn.args[1].args

  fcn_name = esc(signature[1])
  args = esc.(signature[2:end])

  # Check if function body contains a return:
  MacroTools.postwalk(body) do x
    !(@capture(x, return _)) || error("Return statement not permitted in a kernel function $(signature[1])")
  end

  check_kwargs(:makekernel, kwargs...)
  kwargnames = map(t->t[1], map(t->Pair(t.args...), kwargs))
  kwargvals = map(t->t[2],map(t->Pair(t.args...), kwargs))

  idx_fastgtpsa = findfirst(t->t==:fastgtpsa, kwargnames)
  idx_inbounds = findfirst(t->t==:inbounds, kwargnames)

   if isnothing(idx_fastgtpsa) || !kwargvals[idx_fastgtpsa] # no fastgtpsa
    if isnothing(idx_inbounds) || kwargvals[idx_inbounds] # inbounds
      return quote
        @inline function $(fcn_name)($(args...))
          @inbounds begin
            $(body)
          end
        end
      end
    else # no inbounds
      return quote
        @inline function $(fcn_name)($(args...))
          $(body)
        end
      end
    end
  else # fastgtpsa
    if isnothing(idx_inbounds) || kwargvals[idx_inbounds] # inbounds
      return quote
        @inline function $(fcn_name)($(args...))
          @inbounds begin @FastGTPSA begin
            $(body)
          end end
        end
      end
    else # no inbounds
      return quote
        @inline function $(fcn_name)($(args...))
          @FastGTPSA begin
            $(body)
          end 
        end
      end

    end
  end

end
#=

for particle in particles
  for ele in ring

  end
end

for ele in ring
  # do a bunch pre pro
  for particle in particle

  end
end
 =#