abstract type GPUBackendTag end
struct CPUTag <: GPUBackendTag end
struct CUDATag <: GPUBackendTag end
struct ROCmTag <: GPUBackendTag end
struct MetalTag <: GPUBackendTag end
struct OneAPITag <: GPUBackendTag end
struct GenericGPUTag <: GPUBackendTag end

gpu_backend_tag(::Type{T}) where {T<:Number} = CPUTag()
gpu_backend_tag(::CPU) = CPUTag()
gpu_backend_tag(::GPU) = GenericGPUTag()

has_fp64(::GPUBackendTag) = true

@inline _gpu_zero(::Type{T}) where {T} = zero(T)
@inline _gpu_one(::Type{T}) where {T} = one(T)
@inline _pi_typed(::Type{T}) where {T} = T(pi)
@inline _half_typed(::Type{T}) where {T} = T(0.5)

# Scalar Float32 libm in Julia often uses Float64 internally (fpext), which Metal IR rejects.
# FastMath single-precision paths avoid that; SIMD.Vec and Float64 keep the standard libm calls.
@inline _gpu_safe_sin(x::Float32) = Base.FastMath.sin_fast(x)
@inline _gpu_safe_sin(x::Float64) = sin(x)
@inline _gpu_safe_sin(x) = sin(x)

@inline _gpu_safe_cos(x::Float32) = Base.FastMath.cos_fast(x)
@inline _gpu_safe_cos(x::Float64) = cos(x)
@inline _gpu_safe_cos(x) = cos(x)

@inline _gpu_safe_tan(x::Float32) = Base.FastMath.tan_fast(x)
@inline _gpu_safe_tan(x::Float64) = tan(x)
@inline _gpu_safe_tan(x) = tan(x)

@inline _gpu_safe_asin(x::Float32) = Base.FastMath.asin_fast(x)
@inline _gpu_safe_asin(x::Float64) = asin(x)
@inline _gpu_safe_asin(x) = asin(x)

@inline _gpu_safe_sinh(x::Float32) = Base.FastMath.sinh_fast(x)
@inline _gpu_safe_sinh(x::Float64) = sinh(x)
@inline _gpu_safe_sinh(x) = sinh(x)

@inline _gpu_safe_cosh(x::Float32) = Base.FastMath.cosh_fast(x)
@inline _gpu_safe_cosh(x::Float64) = cosh(x)
@inline _gpu_safe_cosh(x) = cosh(x)

@inline _trig_sin(x) = _gpu_safe_sin(x)
@inline _trig_cos(x) = _gpu_safe_cos(x)
@inline _trig_tan(x) = _gpu_safe_tan(x)
@inline _trig_asin(x) = _gpu_safe_asin(x)
@inline _trig_sinh(x) = _gpu_safe_sinh(x)
@inline _trig_cosh(x) = _gpu_safe_cosh(x)

# `Base.cbrt(::Float32)` refines with Float64 (`_improve_cbrt`), which Metal IR rejects.
@inline _cbrt_eps(::Type{Float32}) = 0.060555443f0 # cbrt(eps(Float32))
@inline _cbrt_eps(::Type{Float64}) = cbrt(eps(Float64))
@inline _cbrt_eps(::Type{T}) where {T} = cbrt(eps(T))

@inline _gpu_safe_cbrt(x::Float32) = sign(x) * abs(x)^(1.0f0 / 3.0f0)
@inline _gpu_safe_cbrt(x::Float64) = cbrt(x)
@inline _gpu_safe_cbrt(x) = cbrt(x)

@inline _cbrt_lane_eps(x::Float32) = _cbrt_eps(Float32)
@inline _cbrt_lane_eps(x::Float64) = _cbrt_eps(Float64)
@inline _cbrt_lane_eps(x::SIMD.Vec{N,Float32}) where {N} = SIMD.Vec{N,Float32}(_cbrt_eps(Float32))
@inline _cbrt_lane_eps(x::SIMD.Vec{N,Float64}) where {N} = SIMD.Vec{N,Float64}(_cbrt_eps(Float64))
@inline _cbrt_lane_eps(x) = _gpu_safe_cbrt(_lane_eps(x))

@inline safe_div(a, b) = a / b

@inline gaussian_random(tag::GPUBackendTag, sigma1, sigma2) = _gaussian_random(tag, sigma1, sigma2)
@inline function _gaussian_random(::CPUTag, sigma1, sigma2)
  return randn() * sigma1, randn() * sigma2
end
@inline function _gaussian_random(::GPUBackendTag, sigma1, sigma2)
  s, c = sincospi(2 * rand())
  t = sqrt(-2 * log(rand()))
  return c * t * sigma1, s * t * sigma2
end
