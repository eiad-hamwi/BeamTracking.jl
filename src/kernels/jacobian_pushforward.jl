"""
    Jet6{T}

Dual-style scalar with value `x` and six partials `dx` for coordinate Jacobian pushforward.
`T` should match the orbital coordinate eltype (`eltype(coords.v)`). Arithmetic is defined for
`T <: Number`. SIMD lanes use `T <: SIMD.Vec` for `x`. [`abs`](@ref), [`sign`](@ref), and
ordering (`<`, etc.) delegate to `x` (scalar real or per-lane vector). `Complex` `Jet6` is
rejected for [`abs`](@ref) / [`sign`](@ref).
"""
struct Jet6{T}
  x::T
  dx::SVector{6,T}
end

# High-precision Yoshida coefficients as `Float64`; combine with `ds_step` using its numeric type
# (supports e.g. `Float32`, `Float16`, `Complex`, `Rational`, integer step).
@inline function _yoshida_weight(ds_step, c::Float64)
  return oftype(float(ds_step), c) * ds_step
end

@inline value(a::Jet6) = a.x
@inline value(a::Number) = a
@inline grad(a::Jet6) = a.dx
@inline grad(a::Number) = zero(SVector{6,typeof(a)})

@inline function seed6(x::SVector{6,T}) where {T}
  return ntuple(k -> Jet6(x[k], SVector{6,T}(ntuple(j -> ifelse(j == k, one(T), zero(T)), Val(6)))), Val(6))
end

@inline Base.zero(::Type{Jet6{T}}) where {T} = Jet6(zero(T), zero(SVector{6,T}))
@inline Base.one(::Type{Jet6{T}}) where {T} = Jet6(one(T), zero(SVector{6,T}))
@inline Base.zero(a::Jet6) = zero(typeof(a))
@inline Base.one(a::Jet6) = one(typeof(a))
@inline Base.promote_rule(::Type{Jet6{T}}, ::Type{S}) where {T,S<:Number} = Jet6{promote_type(T,S)}
@inline Base.promote_rule(::Type{Jet6{T}}, ::Type{Jet6{S}}) where {T,S} = Jet6{promote_type(T,S)}
@inline Base.convert(::Type{Jet6{T}}, a::Jet6) where {T} = Jet6(convert(T, a.x), convert(SVector{6,T}, a.dx))
@inline Base.convert(::Type{Jet6{T}}, a::Number) where {T} = Jet6(convert(T, a), zero(SVector{6,T}))

@inline Base.:+(a::Jet6, b::Jet6) = Jet6(a.x + b.x, a.dx + b.dx)
@inline Base.:+(a::Jet6, b::Number) = Jet6(a.x + b, a.dx)
@inline Base.:+(a::Number, b::Jet6) = Jet6(a + b.x, b.dx)
@inline Base.:+(a::Jet6, b::SIMD.Vec) = Jet6(a.x + b, a.dx)
@inline Base.:+(a::SIMD.Vec, b::Jet6) = Jet6(a + b.x, b.dx)
@inline Base.:-(a::Jet6, b::Jet6) = Jet6(a.x - b.x, a.dx - b.dx)
@inline Base.:-(a::Jet6, b::Number) = Jet6(a.x - b, a.dx)
@inline Base.:-(a::Number, b::Jet6) = Jet6(a - b.x, -b.dx)
@inline Base.:-(a::Jet6, b::SIMD.Vec) = Jet6(a.x - b, a.dx)
@inline Base.:-(a::SIMD.Vec, b::Jet6) = Jet6(a - b.x, -b.dx)
@inline Base.:-(a::Jet6) = Jet6(-a.x, -a.dx)
@inline _svec_mul(a, dx) = SVector{6}(ntuple(k -> a * dx[k], Val(6)))
@inline _svec_div(dx, a) = SVector{6}(ntuple(k -> dx[k] / a, Val(6)))
@inline _svec_axpby(a, x, b, y) = SVector{6}(ntuple(k -> a*x[k] + b*y[k], Val(6)))
@inline Base.:*(a::Jet6, b::Jet6) = Jet6(a.x*b.x, _svec_axpby(a.x, b.dx, b.x, a.dx))
@inline Base.:*(a::Jet6, b::Number) = Jet6(a.x*b, _svec_mul(b, a.dx))
@inline Base.:*(a::Number, b::Jet6) = Jet6(a*b.x, _svec_mul(a, b.dx))
@inline Base.:*(a::Jet6, b::SIMD.Vec) = Jet6(a.x*b, _svec_mul(b, a.dx))
@inline Base.:*(a::SIMD.Vec, b::Jet6) = Jet6(a*b.x, _svec_mul(a, b.dx))
@inline Base.:/(a::Jet6, b::Jet6) = Jet6(a.x/b.x, _svec_div(_svec_axpby(b.x, a.dx, -a.x, b.dx), b.x*b.x))
@inline Base.:/(a::Jet6, b::Number) = Jet6(a.x/b, _svec_div(a.dx, b))
@inline Base.:/(a::Number, b::Jet6) = Jet6(a/b.x, _svec_div(_svec_mul(-a, b.dx), b.x*b.x))
@inline Base.:/(a::Jet6, b::SIMD.Vec) = Jet6(a.x/b, _svec_div(a.dx, b))
@inline Base.:/(a::SIMD.Vec, b::Jet6) = Jet6(a/b.x, _svec_div(_svec_mul(-a, b.dx), b.x*b.x))
@inline Base.sqrt(a::Jet6) = (s = sqrt(a.x); Jet6(s, _svec_div(a.dx, 2s)))
@inline Base.sin(a::Jet6) = Jet6(_trig_sin(a.x), _svec_mul(_trig_cos(a.x), a.dx))
@inline Base.cos(a::Jet6) = Jet6(_trig_cos(a.x), _svec_mul(-_trig_sin(a.x), a.dx))
@inline Base.tan(a::Jet6) = (t = _trig_tan(a.x); Jet6(t, _svec_mul(one(a.x) + t * t, a.dx)))
@inline Base.asin(a::Jet6) = Jet6(_trig_asin(a.x), _svec_div(a.dx, sqrt(one(a.x) - a.x * a.x)))
@inline Base.sinh(a::Jet6) = Jet6(_trig_sinh(a.x), _svec_mul(_trig_cosh(a.x), a.dx))
@inline Base.cosh(a::Jet6) = Jet6(_trig_cosh(a.x), _svec_mul(_trig_sinh(a.x), a.dx))
@inline function atan2(y::Jet6, x::Jet6)
  d = x.x*x.x + y.x*y.x
  return Jet6(atan2(y.x, x.x), _svec_div(_svec_axpby(x.x, y.dx, -y.x, x.dx), d))
end
@inline Base.atan(y::Jet6, x::Jet6) = atan2(y, x)
@inline function Base.abs(a::Jet6{<:Complex})
  error("Jacobian pushforward abs(::Jet6{<:Complex}) is not defined; use real coordinates.")
end
@inline Base.abs(a::Jet6) = Jet6(abs(a.x), _svec_mul(sign(a.x), a.dx))
@inline function Base.sign(a::Jet6{<:Complex})
  error("Jacobian pushforward sign(::Jet6{<:Complex}) is not defined; use real coordinates.")
end
@inline Base.sign(a::Jet6) = sign(a.x)
@inline Base.:<(a::Jet6, b::Jet6) = a.x < b.x
@inline Base.:<(a::Jet6, b::Number) = a.x < b
@inline Base.:<(a::Number, b::Jet6) = a < b.x
@inline Base.:>(a::Jet6, b::Jet6) = a.x > b.x
@inline Base.:>(a::Jet6, b::Number) = a.x > b
@inline Base.:>(a::Number, b::Jet6) = a > b.x
@inline Base.:<=(a::Jet6, b::Jet6) = a.x <= b.x
@inline Base.:<=(a::Jet6, b::Number) = a.x <= b
@inline Base.:<=(a::Number, b::Jet6) = a <= b.x
@inline Base.:>=(a::Jet6, b::Jet6) = a.x >= b.x
@inline Base.:>=(a::Jet6, b::Number) = a.x >= b
@inline Base.:>=(a::Number, b::Jet6) = a >= b.x

@inline _mask_select(mask, a, b) = vifelse(mask, a, b)
@inline _mask_select(mask::Bool, a::Jet6, b::Jet6) = ifelse(mask, a, b)
@inline _mask_select(mask, a::Jet6, b::Jet6) = Jet6(vifelse(mask, a.x, b.x), SVector{6}(ntuple(k -> vifelse(mask, a.dx[k], b.dx[k]), Val(6))))
@inline Base.ifelse(mask::SIMD.Vec{N,Bool}, a::Jet6, b::Jet6) where {N} = _mask_select(mask, a, b)

@inline function _lane_eps(x::Number)
  r = real(x)
  return typeof(r) <: AbstractFloat ? eps(r) : eps(float(r))
end
@inline _lane_eps(x::SIMD.Vec{N,T}) where {N,T} = SIMD.Vec{N,T}(eps(T))

@inline function sincu(a::Jet6)
  s = sincu(a.x)
  thr = _cbrt_lane_eps(a.x)
  ds = vifelse(abs(a.x) < thr, zero(a.x), (_trig_cos(a.x) - s) / a.x)
  return Jet6(s, _svec_mul(ds, a.dx))
end

@inline function sinhcu(a::Jet6)
  s = sinhcu(a.x)
  thr = _cbrt_lane_eps(a.x)
  ds = vifelse(abs(a.x) < thr, zero(a.x), (_trig_cosh(a.x) - s) / a.x)
  return Jet6(s, _svec_mul(ds, a.dx))
end

@inline function allocate_coordinate_jacobian(coords::Coords)
  return similar(coords.v, size(coords.v, 1), 6, 6)
end

@inline allocate_coordinate_jacobian(bunch::Bunch) = allocate_coordinate_jacobian(bunch.coords)

function identity_jacobian!(jac)
  size(jac, 2) == 6 && size(jac, 3) == 6 || error("Jacobian sidecar must have shape N x 6 x 6")
  jac .= zero(eltype(jac))
  @inbounds for p in axes(jac, 1), a in 1:6
    jac[p,a,a] = one(eltype(jac))
  end
  return jac
end

@inline function coord_svec(coords::Coords, i)
  v = coords.v
  return SVector(v[i,XI], v[i,PXI], v[i,YI], v[i,PYI], v[i,ZI], v[i,PZI])
end

@inline function store_coord_svec!(coords::Coords, i, x)
  v = coords.v
  v[i,XI] = x[XI]
  v[i,PXI] = x[PXI]
  v[i,YI] = x[YI]
  v[i,PYI] = x[PYI]
  v[i,ZI] = x[ZI]
  v[i,PZI] = x[PZI]
  return nothing
end

@inline function masked_store_coord_svec!(coords::Coords, i, xnew, alive)
  vold = coord_svec(coords, i)
  store_coord_svec!(coords, i, SVector{6}(ntuple(k -> _mask_select(alive, xnew[k], vold[k]), Val(6))))
  return nothing
end

@inline function smat6_from_entries(f, ::Type{T}) where {T}
  return SMatrix{6,6,T}(ntuple(k -> begin
    r = ((k - 1) % 6) + 1
    c = ((k - 1) ÷ 6) + 1
    f(r, c)
  end, Val(36)))
end

@inline function jac_smat6(y)
  T = promote_type(map(a -> eltype(grad(a)), Tuple(y))...)
  return SMatrix{6,6,T}(ntuple(k -> begin
    r = ((k - 1) % 6) + 1
    c = ((k - 1) ÷ 6) + 1
    grad(y[r])[c]
  end, Val(36)))
end

@inline function value_svec6(y)
  return SVector(value(y[1]), value(y[2]), value(y[3]), value(y[4]), value(y[5]), value(y[6]))
end

@inline function map6_value_and_jac(f, x, args...)
  yj = f(SVector{6}(seed6(x)), args...)
  return value_svec6(yj), jac_smat6(yj)
end

@inline function left_compose_jac6!(jac, i, A)
  @inbounds for c in 1:6
    j1 = jac[i,1,c]
    j2 = jac[i,2,c]
    j3 = jac[i,3,c]
    j4 = jac[i,4,c]
    j5 = jac[i,5,c]
    j6 = jac[i,6,c]
    jac[i,1,c] = A[1,1]*j1 + A[1,2]*j2 + A[1,3]*j3 + A[1,4]*j4 + A[1,5]*j5 + A[1,6]*j6
    jac[i,2,c] = A[2,1]*j1 + A[2,2]*j2 + A[2,3]*j3 + A[2,4]*j4 + A[2,5]*j5 + A[2,6]*j6
    jac[i,3,c] = A[3,1]*j1 + A[3,2]*j2 + A[3,3]*j3 + A[3,4]*j4 + A[3,5]*j5 + A[3,6]*j6
    jac[i,4,c] = A[4,1]*j1 + A[4,2]*j2 + A[4,3]*j3 + A[4,4]*j4 + A[4,5]*j5 + A[4,6]*j6
    jac[i,5,c] = A[5,1]*j1 + A[5,2]*j2 + A[5,3]*j3 + A[5,4]*j4 + A[5,5]*j5 + A[5,6]*j6
    jac[i,6,c] = A[6,1]*j1 + A[6,2]*j2 + A[6,3]*j3 + A[6,4]*j4 + A[6,5]*j5 + A[6,6]*j6
  end
  return nothing
end

@inline function left_compose_jac6_masked!(jac, i, A, alive)
  @inbounds for c in 1:6
    j1 = jac[i,1,c]
    j2 = jac[i,2,c]
    j3 = jac[i,3,c]
    j4 = jac[i,4,c]
    j5 = jac[i,5,c]
    j6 = jac[i,6,c]
    n1 = A[1,1]*j1 + A[1,2]*j2 + A[1,3]*j3 + A[1,4]*j4 + A[1,5]*j5 + A[1,6]*j6
    n2 = A[2,1]*j1 + A[2,2]*j2 + A[2,3]*j3 + A[2,4]*j4 + A[2,5]*j5 + A[2,6]*j6
    n3 = A[3,1]*j1 + A[3,2]*j2 + A[3,3]*j3 + A[3,4]*j4 + A[3,5]*j5 + A[3,6]*j6
    n4 = A[4,1]*j1 + A[4,2]*j2 + A[4,3]*j3 + A[4,4]*j4 + A[4,5]*j5 + A[4,6]*j6
    n5 = A[5,1]*j1 + A[5,2]*j2 + A[5,3]*j3 + A[5,4]*j4 + A[5,5]*j5 + A[5,6]*j6
    n6 = A[6,1]*j1 + A[6,2]*j2 + A[6,3]*j3 + A[6,4]*j4 + A[6,5]*j5 + A[6,6]*j6
    jac[i,1,c] = _mask_select(alive, n1, j1)
    jac[i,2,c] = _mask_select(alive, n2, j2)
    jac[i,3,c] = _mask_select(alive, n3, j3)
    jac[i,4,c] = _mask_select(alive, n4, j4)
    jac[i,5,c] = _mask_select(alive, n5, j5)
    jac[i,6,c] = _mask_select(alive, n6, j6)
  end
  return nothing
end

@inline function left_compose_rotation_jac6_masked!(jac, i, alive, w11, w12, w21, w22, a11, a12, a13, a21, a22, a23)
  @inbounds for c in 1:6
    jx = jac[i,XI,c]
    jy = jac[i,YI,c]
    jpx = jac[i,PXI,c]
    jpy = jac[i,PYI,c]
    nx = w11*jx + w12*jy
    ny = w21*jx + w22*jy
    npx = a11*jpx + a12*jpy + a13*jac[i,PZI,c]
    npy = a21*jpx + a22*jpy + a23*jac[i,PZI,c]
    jac[i,XI,c] = _mask_select(alive, nx, jx)
    jac[i,YI,c] = _mask_select(alive, ny, jy)
    jac[i,PXI,c] = _mask_select(alive, npx, jpx)
    jac[i,PYI,c] = _mask_select(alive, npy, jpy)
  end
  return nothing
end

@inline function track_aperture_rectangular_with_jac!(
    i,
    coords::Coords,
    x1,
    x2,
    y1,
    y2
  )

  track_aperture_rectangular!(i, coords, x1, x2, y1, y2)

  return nothing
end


@inline function track_aperture_elliptical_with_jac!(
    i,
    coords::Coords,
    x1,
    x2,
    y1,
    y2
  )

  track_aperture_elliptical!(i, coords, x1, x2, y1, y2)

  return nothing
end

@inline function exact_drift_map6(x, beta_0, gamsqr_0, tilde_m, L)
  X, PX, Y, PY, Z, PZ = x
  P = one(PZ) + PZ
  Pt2 = PX*PX + PY*PY
  Ps = sqrt(P*P - Pt2)
  E = sqrt(P*P + tilde_m*tilde_m)
  Xn = X + PX * L / Ps
  Yn = Y + PY * L / Ps
  Zn = Z - P * L * (one(Ps)/Ps - one(E)/(beta_0*E))
  return SVector(Xn, PX, Yn, PY, Zn, PZ)
end

@inline function exact_drift_with_jac!(i, coords::Coords, beta_0, gamsqr_0, tilde_m, L)
  alive = coords.state[i] == STATE_ALIVE
  x0 = coord_svec(coords, i)
  P = one(x0[PZI]) + x0[PZI]
  Ps2 = P*P - x0[PXI]*x0[PXI] - x0[PYI]*x0[PYI]
  good_momenta = Ps2 > zero(Ps2)
  coords.state[i] = vifelse((!good_momenta) & alive, STATE_LOST, coords.state[i])
  alive = coords.state[i] == STATE_ALIVE
  safe_x = SVector(x0[XI], _mask_select(alive, x0[PXI], zero(x0[PXI])), x0[YI], _mask_select(alive, x0[PYI], zero(x0[PYI])), x0[ZI], _mask_select(alive, x0[PZI], zero(x0[PZI])))
  x1, A = map6_value_and_jac(exact_drift_map6, safe_x, beta_0, gamsqr_0, tilde_m, L)
  masked_store_coord_svec!(coords, i, x1, alive)
  left_compose_jac6_masked!(coords.jac, i, A, alive)
  return nothing
end

@inline function rotation_w(q_inv)
  w11 = 1 - 2*(q_inv[QY]*q_inv[QY] + q_inv[QZ]*q_inv[QZ])
  w12 =     2*(q_inv[QX]*q_inv[QY] - q_inv[QZ]*q_inv[Q0])
  w13 =     2*(q_inv[QX]*q_inv[QZ] + q_inv[QY]*q_inv[Q0])
  w21 =     2*(q_inv[QX]*q_inv[QY] + q_inv[QZ]*q_inv[Q0])
  w22 = 1 - 2*(q_inv[QX]*q_inv[QX] + q_inv[QZ]*q_inv[QZ])
  w23 =     2*(q_inv[QY]*q_inv[QZ] - q_inv[QX]*q_inv[Q0])
  w31 =     2*(q_inv[QX]*q_inv[QZ] - q_inv[QY]*q_inv[Q0])
  w32 =     2*(q_inv[QY]*q_inv[QZ] + q_inv[QX]*q_inv[Q0])
  w33 = 1 - 2*(q_inv[QX]*q_inv[QX] + q_inv[QY]*q_inv[QY])
  return w11, w12, w13, w21, w22, w23, w31, w32, w33
end

@inline function _lane_mask_all(mask::Bool)
  return mask
end
@inline function _lane_mask_all(mask::SIMD.Vec{N,Bool}) where {N}
  return all(Tuple(mask))
end

@inline function rotation_is_noop(q_inv, z_0)
  q_id = Bool(q_inv[Q0] == one(q_inv[Q0])) &&
         Bool(q_inv[QX] == zero(q_inv[QX])) &&
         Bool(q_inv[QY] == zero(q_inv[QY])) &&
         Bool(q_inv[QZ] == zero(q_inv[QZ]))
  zm = z_0 == zero(z_0)
  return q_id && _lane_mask_all(zm)
end

@inline function rotation_with_jac!(i, coords::Coords, q_inv, z_0)
  if rotation_is_noop(q_inv, z_0)
    return nothing
  end
  alive = coords.state[i] == STATE_ALIVE
  v = coords.v
  P = one(v[i,PZI]) + v[i,PZI]
  Ps2 = P*P - v[i,PXI]*v[i,PXI] - v[i,PYI]*v[i,PYI]
  good_momenta = Ps2 > zero(Ps2)
  coords.state[i] = vifelse((!good_momenta) & alive, STATE_LOST, coords.state[i])
  alive = coords.state[i] == STATE_ALIVE
  safe_pz = _mask_select(alive, v[i,PZI], zero(v[i,PZI]))
  safe_px = _mask_select(alive, v[i,PXI], zero(v[i,PXI]))
  safe_py = _mask_select(alive, v[i,PYI], zero(v[i,PYI]))
  P = one(safe_pz) + safe_pz
  Ps2 = P*P - safe_px*safe_px - safe_py*safe_py
  Ps = sqrt(Ps2)
  w11, w12, w13, w21, w22, w23, _w31, _w32, _w33 = rotation_w(q_inv)
  x0 = v[i,XI]
  y0 = v[i,YI]
  px0 = safe_px
  py0 = safe_py
  v[i,XI] = _mask_select(alive, w11*x0 + w12*y0 + w13*z_0, x0)
  v[i,YI] = _mask_select(alive, w21*x0 + w22*y0 + w23*z_0, y0)
  v[i,PXI] = _mask_select(alive, w11*px0 + w12*py0 + w13*Ps, v[i,PXI])
  v[i,PYI] = _mask_select(alive, w21*px0 + w22*py0 + w23*Ps, v[i,PYI])
  T = typeof(Ps)
  tw11 = T(w11); tw12 = T(w12); tw13 = T(w13)
  tw21 = T(w21); tw22 = T(w22); tw23 = T(w23)
  a11 = tw11 - tw13*px0/Ps
  a12 = tw12 - tw13*py0/Ps
  a13 = tw13*P/Ps
  a21 = tw21 - tw23*px0/Ps
  a22 = tw22 - tw23*py0/Ps
  a23 = tw23*P/Ps
  left_compose_rotation_jac6_masked!(coords.jac, i, alive, tw11, tw12, tw21, tw22, a11, a12, a13, a21, a22, a23)
  return nothing
end

@inline function patch_mark_bad_momenta!(i, coords::Coords)
  @inbounds begin
    v = coords.v
    rel_p = one(v[i,PZI]) + v[i,PZI]
    ps2 = rel_p*rel_p - v[i,PXI]*v[i,PXI] - v[i,PYI]*v[i,PYI]
    good_momenta = ps2 > zero(ps2)
    alive_at_start = coords.state[i] == STATE_ALIVE
    coords.state[i] = vifelse((!good_momenta) & alive_at_start, STATE_LOST, coords.state[i])
  end
  return nothing
end


@inline function patch_offset_with_jac!(i, coords::Coords, tilde_m, dx, dy, dt)
  @inbounds begin
    v = coords.v
    jac = coords.jac

    alive = coords.state[i] == STATE_ALIVE

    x0 = v[i,XI]
    y0 = v[i,YI]
    z0 = v[i,ZI]
    pz0 = v[i,PZI]

    safe_pz = _mask_select(alive, pz0, zero(pz0))

    P = one(safe_pz) + safe_pz
    m2 = tilde_m * tilde_m
    E = sqrt(P*P + m2)

    k = C_LIGHT * dt

    v[i,XI] = _mask_select(alive, x0 - dx, x0)
    v[i,YI] = _mask_select(alive, y0 - dy, y0)
    v[i,ZI] = _mask_select(alive, z0 + P/E * k, z0)

    # d/dPZ [ P / sqrt(P^2 + m^2) ] = m^2 / E^3
    dZ_dPZ = k * m2 / (E*E*E)

    for c in 1:6
      jz = jac[i,ZI,c]
      jpz = jac[i,PZI,c]
      jac[i,ZI,c] = _mask_select(alive, jz + dZ_dPZ * jpz, jz)
    end
  end

  return nothing
end


@inline function patch_final_z_no_rotation_with_jac!(i, coords::Coords, D)
  @inbounds begin
    v = coords.v
    jac = coords.jac

    alive = coords.state[i] == STATE_ALIVE

    px0 = v[i,PXI]
    py0 = v[i,PYI]
    pz0 = v[i,PZI]

    safe_px = _mask_select(alive, px0, zero(px0))
    safe_py = _mask_select(alive, py0, zero(py0))
    safe_pz = _mask_select(alive, pz0, zero(pz0))

    P = one(safe_pz) + safe_pz
    Pt2 = safe_px*safe_px + safe_py*safe_py
    Ps = sqrt(P*P - Pt2)

    z0 = v[i,ZI]
    v[i,ZI] = _mask_select(alive, z0 - D * P/Ps, z0)

    invPs3 = one(Ps) / (Ps*Ps*Ps)

    # f = P/Ps
    # df/dPX = P*PX/Ps^3
    # df/dPY = P*PY/Ps^3
    # df/dPZ = -Pt2/Ps^3
    dZ_dPX = -D * P * safe_px * invPs3
    dZ_dPY = -D * P * safe_py * invPs3
    dZ_dPZ =  D * Pt2 * invPs3

    for c in 1:6
      jz = jac[i,ZI,c]
      jpx = jac[i,PXI,c]
      jpy = jac[i,PYI,c]
      jpz = jac[i,PZI,c]

      nz = jz + dZ_dPX*jpx + dZ_dPY*jpy + dZ_dPZ*jpz
      jac[i,ZI,c] = _mask_select(alive, nz, jz)
    end
  end

  return nothing
end


@inline function add_exact_drift_sf_dependence_jac6_masked!(i, coords::Coords, beta_0, tilde_m, dsf)
  @inbounds begin
    v = coords.v
    jac = coords.jac

    alive = coords.state[i] == STATE_ALIVE

    px0 = v[i,PXI]
    py0 = v[i,PYI]
    pz0 = v[i,PZI]

    safe_px = _mask_select(alive, px0, zero(px0))
    safe_py = _mask_select(alive, py0, zero(py0))
    safe_pz = _mask_select(alive, pz0, zero(pz0))

    P = one(safe_pz) + safe_pz
    Ps = sqrt(P*P - safe_px*safe_px - safe_py*safe_py)
    E = sqrt(P*P + tilde_m*tilde_m)

    # exact_drift_map6 derivatives wrt its length argument L:
    #
    # Xn = X + PX*L/Ps
    # Yn = Y + PY*L/Ps
    # Zn = Z - P*L*(1/Ps - 1/(beta_0*E))
    dX_dL = safe_px / Ps
    dY_dL = safe_py / Ps
    dZ_dL = -P * (one(Ps)/Ps - one(E)/(beta_0*E))

    # In patch!, the drift length is Ld = -s_f,
    # so dLd = -ds_f.
    for c in 1:6
      dLd = -dsf[c]

      jx = jac[i,XI,c]
      jy = jac[i,YI,c]
      jz = jac[i,ZI,c]

      nx = jx + dX_dL * dLd
      ny = jy + dY_dL * dLd
      nz = jz + dZ_dL * dLd

      jac[i,XI,c] = _mask_select(alive, nx, jx)
      jac[i,YI,c] = _mask_select(alive, ny, jy)
      jac[i,ZI,c] = _mask_select(alive, nz, jz)
    end
  end

  return nothing
end


@inline function patch_final_z_rotation_with_jac!(i, coords::Coords, tilde_m, s_f, L, dsf)
  @inbounds begin
    v = coords.v
    jac = coords.jac

    alive = coords.state[i] == STATE_ALIVE

    pz0 = v[i,PZI]
    safe_pz = _mask_select(alive, pz0, zero(pz0))

    P = one(safe_pz) + safe_pz
    m2 = tilde_m * tilde_m
    E = sqrt(P*P + m2)

    sqrt1pm2 = sqrt(one(P) + m2)

    # K = P * sqrt((1 + m^2)/(P^2 + m^2))
    K = P * sqrt1pm2 / E

    # dK/dPZ = sqrt(1 + m^2) * m^2 / E^3
    dK_dPZ = sqrt1pm2 * m2 / (E*E*E)

    sLp = s_f + L

    z0 = v[i,ZI]
    v[i,ZI] = _mask_select(alive, z0 + sLp * K, z0)

    for c in 1:6
      jz = jac[i,ZI,c]
      jpz = jac[i,PZI,c]

      nz = jz + K * dsf[c] + sLp * dK_dPZ * jpz
      jac[i,ZI,c] = _mask_select(alive, nz, jz)
    end
  end

  return nothing
end


# The rotation_with_jac! you pasted does not update coords.q,
# whereas rotation! does. Keep this helper if patch_with_jac!
# should exactly match patch! when coords.q is present.
@inline function rotate_quaternion_only_masked!(i, coords::Coords, q_inv)
  @inbounds begin
    q1 = coords.q

    if !isnothing(q1)
      alive = coords.state[i] == STATE_ALIVE

      q = quat_mul(q_inv, q1[i,Q0], q1[i,QX], q1[i,QY], q1[i,QZ])

      q0 = _mask_select(alive, q[Q0], q1[i,Q0])
      qx = _mask_select(alive, q[QX], q1[i,QX])
      qy = _mask_select(alive, q[QY], q1[i,QY])
      qz = _mask_select(alive, q[QZ], q1[i,QZ])

      q1[i,Q0], q1[i,QX], q1[i,QY], q1[i,QZ] = q0, qx, qy, qz
    end
  end

  return nothing
end


@inline function patch_with_jac!(
    i,
    coords::Coords,
    beta_0,
    gamsqr_0,
    tilde_m,
    dt,
    dx,
    dy,
    dz,
    winv,
    L
  )

  @inbounds begin
    v = coords.v

    # Same initial momentum validity check as patch!.
    # This is needed before patch_offset_with_jac!, since patch_offset!
    # is masked by alive state but does not itself mark bad momenta lost.
    patch_mark_bad_momenta!(i, coords)

    if isnothing(winv)
      patch_offset_with_jac!(i, coords, tilde_m, dx, dy, dt)

      exact_drift_with_jac!(i, coords, beta_0, gamsqr_0, tilde_m, L)

      # patch! does:
      #   z -= (dz - L) * rel_p / ps_0
      #
      # In this branch momenta are unchanged by patch_offset! and exact_drift!,
      # so current momenta give the same rel_p/ps_0.
      patch_final_z_no_rotation_with_jac!(i, coords, dz - L)

    else
      patch_offset_with_jac!(i, coords, tilde_m, dx, dy, dt)

      w31 = 2*(winv[QX]*winv[QZ] - winv[QY]*winv[Q0])
      w32 = 2*(winv[QY]*winv[QZ] + winv[QX]*winv[Q0])
      w33 = 1 - 2*(winv[QX]*winv[QX] + winv[QY]*winv[QY])

      # s_f is computed after patch_offset! and before rotation!, as in patch!.
      s_f = w31*v[i,XI] + w32*v[i,YI] - w33*dz

      # Store global derivative of s_f at this point:
      #
      #   ds_f = w31*dX + w32*dY
      #
      # This must be saved before rotation_with_jac! mutates coords.jac.
      dsf = ntuple(c -> w31*coords.jac[i,XI,c] + w32*coords.jac[i,YI,c], Val(6))

      rotation_with_jac!(i, coords, winv, -dz)

      # Keep this if rotation_with_jac! itself does not update coords.q.
      # rotate_quaternion_only_masked!(i, coords, winv)

      # First apply the ordinary constant-length drift Jacobian,
      # treating -s_f as a frozen scalar.
      exact_drift_with_jac!(i, coords, beta_0, gamsqr_0, tilde_m, -s_f)

      # Then add the missing chain-rule piece from Ld = -s_f(x).
      add_exact_drift_sf_dependence_jac6_masked!(i, coords, beta_0, tilde_m, dsf)

      # patch! does:
      #   z += (s_f + L) * rel_p *
      #        sqrt((1 + tilde_m^2)/(rel_p^2 + tilde_m^2))
      #
      # This also depends on s_f, so add both ds_f and d/dPZ terms.
      patch_final_z_rotation_with_jac!(i, coords, tilde_m, s_f, L, dsf)
    end
  end

  return nothing
end

@inline function linear_bend_fringe_with_jac!(i, coords::Coords, a, tilde_m, Ksol, Kn0, e, sign)
  alive = coords.state[i] == STATE_ALIVE
  v = coords.v
  f = Kn0 * _trig_tan(e)
  v[i,PXI] = _mask_select(alive, v[i,PXI] + f*v[i,XI], v[i,PXI])
  v[i,PYI] = _mask_select(alive, v[i,PYI] - f*v[i,YI], v[i,PYI])
  T = typeof(f)
  A = smat6_from_entries(T) do r, c
    if r == c
      one(T)
    elseif r == PXI && c == XI
      f
    elseif r == PYI && c == YI
      -f
    else
      zero(T)
    end
  end
  left_compose_jac6_masked!(coords.jac, i, A, alive)
  return nothing
end

@inline function multipole_kick_map6(x, ms, knl, ksl, excluding)
  X, PX, Y, PY, Z, PZ = x
  bx, by = normalized_field_runtime(ms, knl, ksl, X, Y, excluding)
  return SVector(X, PX - by, Y, PY + bx, Z, PZ)
end

@inline function normalized_field_runtime(ms, knl, ksl, x, y, excluding)
  N = length(ms)
  z = zero(x*y + first(knl) + first(ksl))
  addN = (ms[N] != excluding && ms[N] > 0)
  by = addN ? knl[N] + z : z
  bx = addN ? ksl[N] + z : z

  for j in (N-1):-1:1
    curknl = knl[j] + z
    curksl = ksl[j] + z
    for m in (ms[j+1]-1):-1:max(ms[j], 1)
      t = (by*x - bx*y) / m
      bx = (by*y + bx*x) / m
      by = t
      if m == ms[j]
        if ms[j] != excluding
          by += curknl
          bx += curksl
        end
      end
    end
  end

  for m in (ms[1]-1):-1:1
    t = (by*x - bx*y) / m
    bx = (by*y + bx*x) / m
    by = t
  end
  return bx, by
end

@inline function multipole_kick_with_jac!(i, coords::Coords, ms, knl, ksl, excluding)
  alive = coords.state[i] == STATE_ALIVE
  x1, A = map6_value_and_jac(multipole_kick_map6, coord_svec(coords, i), ms, knl, ksl, excluding)
  masked_store_coord_svec!(coords, i, x1, alive)
  left_compose_jac6_masked!(coords.jac, i, A, alive)
  return nothing
end

@inline function multipole_and_spin_kick_with_jac!(i, coords::Coords, mm, kn, ks, a, tilde_m, L)
  return multipole_kick_with_jac!(i, coords, mm, kn .* L, ks .* L, 0)
end

@inline function quadrupole_kick_map6(x, beta_0, gamsqr_0, tilde_m, s)
  X, PX, Y, PY, Z, PZ = x
  P = one(PZ) + PZ
  PtSqr = PX*PX + PY*PY
  Ps = sqrt(P*P - PtSqr)
  Xn = X + s * PX * PtSqr / (P * Ps * (P + Ps))
  Yn = Y + s * PY * PtSqr / (P * Ps * (P + Ps))
  E = sqrt(P*P + tilde_m*tilde_m)
  Zn = Z - s * (
    P * (PtSqr - PZ*(2 + PZ)/gamsqr_0) /
    (beta_0 * E * Ps * (beta_0 * E + Ps)) -
    PtSqr / (2 * P*P)
  )
  return SVector(Xn, PX, Yn, PY, Zn, PZ)
end

@inline function quadrupole_kick_with_jac!(i, coords::Coords, beta_0, gamsqr_0, tilde_m, s)
  alive = coords.state[i] == STATE_ALIVE
  x0 = coord_svec(coords, i)
  P = one(x0[PZI]) + x0[PZI]
  Ps2 = P*P - x0[PXI]*x0[PXI] - x0[PYI]*x0[PYI]
  good_momenta = Ps2 > zero(Ps2)
  coords.state[i] = vifelse((!good_momenta) & alive, STATE_LOST, coords.state[i])
  alive = coords.state[i] == STATE_ALIVE
  safe_x = SVector(x0[XI], _mask_select(alive, x0[PXI], zero(x0[PXI])), x0[YI], _mask_select(alive, x0[PYI], zero(x0[PYI])), x0[ZI], _mask_select(alive, x0[PZI], zero(x0[PZI])))
  x1, A = map6_value_and_jac(quadrupole_kick_map6, safe_x, beta_0, gamsqr_0, tilde_m, s)
  masked_store_coord_svec!(coords, i, x1, alive)
  left_compose_jac6_masked!(coords.jac, i, A, alive)
  return nothing
end

@inline function quadrupole_matrix_map6(x, k1, s)
  X, PX, Y, PY, Z, PZ = x
  focus = k1 >= 0
  P = one(PZ) + PZ
  xp = PX / P
  yp = PY / P
  sqrtks = sqrt(abs(k1 / P)) * s
  cosine = _trig_cos(sqrtks)
  coshine = _trig_cosh(sqrtks)
  sinecu = sincu(sqrtks)
  shinecu = sinhcu(sqrtks)
  cx = ifelse(focus, cosine, coshine)
  cy = ifelse(focus, coshine, cosine)
  sx = ifelse(focus, sinecu, shinecu)
  sy = ifelse(focus, shinecu, sinecu)
  PXn = PX * cx - k1 * s * X * sx
  PYn = PY * cy + k1 * s * Y * sy
  Zn = Z - (s / 4) * (
    xp*xp * (1 + sx * cx) +
    yp*yp * (1 + sy * cy) +
    k1 / P * (X*X * (1 - sx * cx) - Y*Y * (1 - sy * cy))
  ) + sign(k1) * (X * xp * (sqrtks * sx)*(sqrtks * sx) - Y * yp * (sqrtks * sy)*(sqrtks * sy)) / 2
  Xn = X * cx + xp * s * sx
  Yn = Y * cy + yp * s * sy
  return SVector(Xn, PXn, Yn, PYn, Zn, PZ)
end

@inline function quadrupole_matrix_with_jac!(i, coords::Coords, k1, s)
  alive = coords.state[i] == STATE_ALIVE
  x1, A = map6_value_and_jac(quadrupole_matrix_map6, coord_svec(coords, i), k1, s)
  masked_store_coord_svec!(coords, i, x1, alive)
  left_compose_jac6_masked!(coords.jac, i, A, alive)
  return nothing
end

@inline function exact_bend_cond(x, theta, g, Kn0, L)
  X, PX, Y, PY, Z, PZ = x
  P = one(PZ) + PZ
  pt = sqrt(P*P - PY*PY)
  arg = PX / pt
  phi1 = theta + _trig_asin(arg)
  gp = Kn0 / pt
  h = one(X) + g*X
  cplus = _trig_cos(phi1)
  splus = _trig_sin(phi1)
  sinc_theta = sincu(theta)
  sgn = sign(L)
  alpha_helper = h*L*sinc_theta
  alpha = 2*h*splus*L*sinc_theta - gp*alpha_helper*alpha_helper
  cond = cplus*cplus + gp*alpha
  return arg, cond, sgn
end

@inline function exact_bend_map6(x, theta, g, Kn0, tilde_m, beta_0, L)
  X, PX, Y, PY, Z, PZ = x
  P = one(PZ) + PZ
  pt = sqrt(P*P - PY*PY)
  arg = PX / pt
  phi1 = theta + _trig_asin(arg)
  gp = Kn0 / pt
  h = one(X) + g*X
  cplus = _trig_cos(phi1)
  splus = _trig_sin(phi1)
  sinc_theta = sincu(theta)
  sinc_theta_2 = sincu(theta/2)
  cosc_theta = sinc_theta_2*sinc_theta_2/2
  sgn = sign(L)
  alpha_helper = h*L*sinc_theta
  alpha = 2*h*splus*L*sinc_theta - gp*alpha_helper*alpha_helper
  cond = cplus*cplus + gp*alpha
  nasty_sqrt = sqrt(cond)
  gp_safe = ifelse(abs(gp) > zero(gp), gp, one(gp))
  pos_cplus = cplus > 0
  xi1 = alpha/(nasty_sqrt + cplus)
  xi2 = (nasty_sqrt - cplus)/gp_safe
  xi = ifelse(!(abs(gp) > zero(gp)) | pos_cplus, xi1, xi2)
  Lcv = -sgn*(L*sinc_theta + X*_trig_sin(theta))
  negative_Lcv = -Lcv
  thetap = 2*(phi1 - sgn*atan2(xi, negative_Lcv))
  Lp = sgn*sqrt(Lcv*Lcv + xi*xi)/sincu(thetap/2)
  Xn = X*_trig_cos(theta) - L*L*g*cosc_theta + xi
  PXn = pt*_trig_sin(phi1 - thetap)
  Yn = Y + PY*Lp/pt
  Zn = Z - P*Lp/pt + L*P/sqrt(tilde_m*tilde_m + P*P)/beta_0
  return SVector(Xn, PXn, Yn, PY, Zn, PZ)
end

@inline function exact_bend_with_jac!(i, coords::Coords, theta, g, Kn0, tilde_m, beta_0, L)
  alive = coords.state[i] == STATE_ALIVE
  x0 = coord_svec(coords, i)
  P = one(x0[PZI]) + x0[PZI]
  pt2 = P*P - x0[PYI]*x0[PYI]
  good_pt = pt2 > zero(pt2)
  coords.state[i] = vifelse((!good_pt) & alive, STATE_LOST, coords.state[i])
  alive = coords.state[i] == STATE_ALIVE
  arg, cond, _sgn = exact_bend_cond(x0, theta, g, Kn0, L)
  good_arg = abs(arg) < one(arg)
  good_cond = cond > zero(cond)
  coords.state[i] = vifelse((!(good_arg & good_cond)) & alive, STATE_LOST, coords.state[i])
  alive = coords.state[i] == STATE_ALIVE
  safe_x = SVector(x0[XI], _mask_select(alive, x0[PXI], zero(x0[PXI])), x0[YI], _mask_select(alive, x0[PYI], zero(x0[PYI])), x0[ZI], _mask_select(alive, x0[PZI], zero(x0[PZI])))
  x1, A = map6_value_and_jac(exact_bend_map6, safe_x, theta, g, Kn0, tilde_m, beta_0, L)
  masked_store_coord_svec!(coords, i, x1, alive)
  left_compose_jac6_masked!(coords.jac, i, A, alive)
  return nothing
end

@inline function exact_curved_drift_with_jac!(i, coords::Coords, e1, e2, g, w, w_inv, a, tilde_m, beta_0, L) 
  rotation_with_jac!(i, coords, w, 0)
  exact_bend_with_jac!(i, coords, g*L, g, 0, tilde_m, beta_0, L)
  rotation_with_jac!(i, coords, w_inv, 0)
end

@inline function dkd_multipole_with_jac!(i, coords::Coords, q, mc2, radiation_damping, beta_0, gamsqr_0, tilde_m, a, mm, kn, ks, L)
  exact_drift_with_jac!(i, coords, beta_0, gamsqr_0, tilde_m, L / 2)
  multipole_and_spin_kick_with_jac!(i, coords, mm, kn, ks, a, tilde_m, L)
  exact_drift_with_jac!(i, coords, beta_0, gamsqr_0, tilde_m, L / 2)
  return nothing
end

@inline function bkb_multipole_with_jac!(i, coords::Coords, q, mc2, radiation_damping, tilde_m, beta_0, a, g, w, w_inv, k0, mm, kn, ks, L)
  knl = kn .* L ./ 2
  ksl = ks .* L ./ 2
  rotation_with_jac!(i, coords, w, zero(coords.v[i,XI]))
  multipole_kick_with_jac!(i, coords, mm, knl, ksl, 1)
  exact_bend_with_jac!(i, coords, g*L, g, k0, tilde_m, beta_0, L)
  multipole_kick_with_jac!(i, coords, mm, knl, ksl, 1)
  rotation_with_jac!(i, coords, w_inv, zero(coords.v[i,XI]))
  return nothing
end

@inline function mkm_quadrupole_with_jac!(i, coords::Coords, q, mc2, radiation_damping, beta_0, gamsqr_0, tilde_m, a, w, w_inv, k1, mm, kn, ks, L)
  alive = coords.state[i] == STATE_ALIVE
  x0 = coord_svec(coords, i)
  P = one(x0[PZI]) + x0[PZI]
  Ps2 = P*P - x0[PXI]*x0[PXI] - x0[PYI]*x0[PYI]
  coords.state[i] = vifelse((!(Ps2 > zero(Ps2))) & alive, STATE_LOST, coords.state[i])
  knl = kn .* L ./ 2
  ksl = ks .* L ./ 2
  multipole_kick_with_jac!(i, coords, mm, knl, ksl, 2)
  quadrupole_kick_with_jac!(i, coords, beta_0, gamsqr_0, tilde_m, L / 2)
  rotation_with_jac!(i, coords, w, zero(coords.v[i,XI]))
  quadrupole_matrix_with_jac!(i, coords, k1, L)
  rotation_with_jac!(i, coords, w_inv, zero(coords.v[i,XI]))
  quadrupole_kick_with_jac!(i, coords, beta_0, gamsqr_0, tilde_m, L / 2)
  multipole_kick_with_jac!(i, coords, mm, knl, ksl, 2)
  return nothing
end

@inline function order_two_integrator_with_jac!(i, coords::Coords, ker, params, photon_params, ds_step, num_steps, edge_params, ::Val{fringe_in}, ::Val{fringe_out}, L) where {fringe_in,fringe_out}
  if !isnothing(edge_params) && fringe_in
    a, tilde_m, Ksol, Kn0, e1, e2 = edge_params
    linear_bend_fringe_with_jac!(i, coords, a, tilde_m, Ksol, Kn0, e1, 1)
  end
  for step in 1:num_steps
    ker(i, coords, params..., ds_step)
  end
  if !isnothing(edge_params) && fringe_out
    a, tilde_m, Ksol, Kn0, e1, e2 = edge_params
    linear_bend_fringe_with_jac!(i, coords, a, tilde_m, Ksol, Kn0, e2, -1)
  end
  return nothing
end

@inline function order_four_integrator_with_jac!(i, coords::Coords, ker, params, photon_params, ds_step, num_steps, edge_params, ::Val{fringe_in}, ::Val{fringe_out}, L) where {fringe_in,fringe_out}
  w0 = _yoshida_weight(ds_step, -1.7024143839193153215916254339390434324741363525390625)
  w1 = _yoshida_weight(ds_step, 1.3512071919596577718181151794851757586002349853515625)
  if !isnothing(edge_params) && fringe_in
    a, tilde_m, Ksol, Kn0, e1, e2 = edge_params
    linear_bend_fringe_with_jac!(i, coords, a, tilde_m, Ksol, Kn0, e1, 1)
  end
  for step in 1:num_steps
    ker(i, coords, params..., w1)
    ker(i, coords, params..., w0)
    ker(i, coords, params..., w1)
  end
  if !isnothing(edge_params) && fringe_out
    a, tilde_m, Ksol, Kn0, e1, e2 = edge_params
    linear_bend_fringe_with_jac!(i, coords, a, tilde_m, Ksol, Kn0, e2, -1)
  end
  return nothing
end

@inline function order_six_integrator_with_jac!(i, coords::Coords, ker, params, photon_params, ds_step, num_steps, edge_params, ::Val{fringe_in}, ::Val{fringe_out}, L) where {fringe_in,fringe_out}
  w0 = _yoshida_weight(ds_step, 1.315186320683911169737712043570355)
  w1 = _yoshida_weight(ds_step, -1.17767998417887100694641568096432)
  w2 = _yoshida_weight(ds_step, 0.235573213359358133684793182978535)
  w3 = _yoshida_weight(ds_step, 0.784513610477557263819497633866351)
  if !isnothing(edge_params) && fringe_in
    a, tilde_m, Ksol, Kn0, e1, e2 = edge_params
    linear_bend_fringe_with_jac!(i, coords, a, tilde_m, Ksol, Kn0, e1, 1)
  end
  for step in 1:num_steps
    ker(i, coords, params..., w3)
    ker(i, coords, params..., w2)
    ker(i, coords, params..., w1)
    ker(i, coords, params..., w0)
    ker(i, coords, params..., w1)
    ker(i, coords, params..., w2)
    ker(i, coords, params..., w3)
  end
  if !isnothing(edge_params) && fringe_out
    a, tilde_m, Ksol, Kn0, e1, e2 = edge_params
    linear_bend_fringe_with_jac!(i, coords, a, tilde_m, Ksol, Kn0, e2, -1)
  end
  return nothing
end

@inline function order_eight_integrator_with_jac!(i, coords::Coords, ker, params, photon_params, ds_step, num_steps, edge_params, ::Val{fringe_in}, ::Val{fringe_out}, L) where {fringe_in,fringe_out}
  w0 = _yoshida_weight(ds_step, 1.7084530707869978)
  w1 = _yoshida_weight(ds_step, 0.102799849391985)
  w2 = _yoshida_weight(ds_step, -1.96061023297549)
  w3 = _yoshida_weight(ds_step, 1.93813913762276)
  w4 = _yoshida_weight(ds_step, -0.158240635368243)
  w5 = _yoshida_weight(ds_step, -1.44485223686048)
  w6 = _yoshida_weight(ds_step, 0.253693336566229)
  w7 = _yoshida_weight(ds_step, 0.914844246229740)
  if !isnothing(edge_params) && fringe_in
    a, tilde_m, Ksol, Kn0, e1, e2 = edge_params
    linear_bend_fringe_with_jac!(i, coords, a, tilde_m, Ksol, Kn0, e1, 1)
  end
  for step in 1:num_steps
    ker(i, coords, params..., w7)
    ker(i, coords, params..., w6)
    ker(i, coords, params..., w5)
    ker(i, coords, params..., w4)
    ker(i, coords, params..., w3)
    ker(i, coords, params..., w2)
    ker(i, coords, params..., w1)
    ker(i, coords, params..., w0)
    ker(i, coords, params..., w1)
    ker(i, coords, params..., w2)
    ker(i, coords, params..., w3)
    ker(i, coords, params..., w4)
    ker(i, coords, params..., w5)
    ker(i, coords, params..., w6)
    ker(i, coords, params..., w7)
  end
  if !isnothing(edge_params) && fringe_out
    a, tilde_m, Ksol, Kn0, e1, e2 = edge_params
    linear_bend_fringe_with_jac!(i, coords, a, tilde_m, Ksol, Kn0, e2, -1)
  end
  return nothing
end

@inline function check_jacobian_sidecar(coords::Coords)
  jac = coords.jac
  jac === nothing && error("Internal error: Jacobian preflight requires coords.jac")
  size(jac) == (size(coords.v, 1), 6, 6) || error("Jacobian sidecar must have shape (N, 6, 6), got $(size(jac)) for N=$(size(coords.v, 1))")
  eltype(jac) == eltype(coords.v) || error("Jacobian sidecar eltype must match coords.v eltype")
  isnothing(coords.q) || error("Jacobian pushforward does not support spin coordinates")
  isempty(coords.callbacks) || error("Jacobian pushforward does not support callbacks")
  return nothing
end

@inline function _preflight_jacobian_kcall(kcall::KernelCall)
  if kcall.kernel === dkd_multipole_with_jac! || kcall.kernel === bkb_multipole_with_jac! || kcall.kernel === mkm_quadrupole_with_jac!
    kcall.args[3] == false || error("Jacobian pushforward does not support radiation damping")
  elseif kcall.kernel === order_two_integrator_with_jac! || kcall.kernel === order_four_integrator_with_jac! ||
         kcall.kernel === order_six_integrator_with_jac! || kcall.kernel === order_eight_integrator_with_jac!
    isnothing(kcall.args[3]) || error("Jacobian pushforward does not support stochastic radiation")
  end
  return nothing
end

@inline function preflight_jacobian_tracking(coords::Coords, kc::KernelChain; use_KA::Bool, use_explicit_SIMD::Bool)
  check_jacobian_sidecar(coords)
  jac = coords.jac
  use_KA && use_explicit_SIMD && error("Cannot use both KernelAbstractions (KA) and explicit SIMD")
  for kcall in kc.chain
    _preflight_jacobian_kcall(kcall)
  end
  if use_KA
    get_backend(jac) == get_backend(coords.v) || error("Jacobian sidecar backend must match coords.v backend")
  end
  return nothing
end

@inline blank_kernel_with_jac!(i, coords, args...) = nothing

@inline _jacobian_kernel(::typeof(blank_kernel!)) = blank_kernel_with_jac!
@inline _jacobian_kernel(::typeof(exact_drift!)) = exact_drift_with_jac!
@inline _jacobian_kernel(::typeof(exact_curved_drift!)) = exact_curved_drift_with_jac!
@inline _jacobian_kernel(::typeof(rotation!)) = rotation_with_jac!
@inline _jacobian_kernel(::typeof(patch!)) = patch_with_jac!
@inline _jacobian_kernel(::typeof(track_aperture_rectangular!)) = track_aperture_rectangular_with_jac!
@inline _jacobian_kernel(::typeof(track_aperture_elliptical!)) = track_aperture_elliptical_with_jac!
@inline _jacobian_kernel(::typeof(linear_bend_fringe!)) = linear_bend_fringe_with_jac!
@inline _jacobian_kernel(::typeof(multipole_kick!)) = multipole_kick_with_jac!
@inline _jacobian_kernel(::typeof(quadrupole_kick!)) = quadrupole_kick_with_jac!
@inline _jacobian_kernel(::typeof(quadrupole_matrix!)) = quadrupole_matrix_with_jac!
@inline _jacobian_kernel(::typeof(exact_bend!)) = exact_bend_with_jac!
@inline _jacobian_kernel(::typeof(dkd_multipole!)) = dkd_multipole_with_jac!
@inline _jacobian_kernel(::typeof(bkb_multipole!)) = bkb_multipole_with_jac!
@inline _jacobian_kernel(::typeof(mkm_quadrupole!)) = mkm_quadrupole_with_jac!
@inline _jacobian_kernel(::typeof(order_two_integrator!)) = order_two_integrator_with_jac!
@inline _jacobian_kernel(::typeof(order_four_integrator!)) = order_four_integrator_with_jac!
@inline _jacobian_kernel(::typeof(order_six_integrator!)) = order_six_integrator_with_jac!
@inline _jacobian_kernel(::typeof(order_eight_integrator!)) = order_eight_integrator_with_jac!
@inline _jacobian_kernel(f) = error("Jacobian pushforward does not support kernel $(f)")

@inline _jacobianized_args(kernel, args) = args

@inline function _jacobianized_args(kernel::Union{
  typeof(order_two_integrator!),
  typeof(order_four_integrator!),
  typeof(order_six_integrator!),
  typeof(order_eight_integrator!)
}, args)
  return (_jacobian_kernel(args[1]), Base.tail(args)...)
end

@inline function jacobianize(kcall::KernelCall)
  return KernelCall(_jacobian_kernel(kcall.kernel), _jacobianized_args(kcall.kernel, kcall.args))
end

@inline function jacobianize(kc::KernelChain)
  return KernelChain(map(jacobianize, kc.chain), kc.ref)
end

"""
    track!(bunch::Bunch, kcall_or_chain; kwargs...)

Track `bunch.coords` through a [`KernelCall`](@ref) or [`KernelChain`](@ref).

Uses [`launch!`](@ref): if `coords.jac === nothing` runs the primal path; otherwise runs the
coordinate Jacobian pushforward using `coords.jac` (see [`Bunch`](@ref) keyword `jacobian` / `jac`).

Returns `bunch`.
"""
function track!(bunch::Bunch, kcall_or_chain; kwargs...)
  launch!(bunch.coords, kcall_or_chain; kwargs...)
  return bunch
end

function value_and_jacobian(
  x0,
  kcall_or_chain;
  species=Species(),
  p_over_q_ref=NaN,
  t_ref=0,
  kwargs...
)
  length(x0) == 6 || error("value_and_jacobian expects a 6-vector of orbital coordinates")
  v = reshape(copy(collect(x0)), 1, 6)
  bunch = Bunch(v=v, species=species, p_over_q_ref=p_over_q_ref, t_ref=t_ref, jacobian=true)
  track!(bunch, kcall_or_chain; kwargs...)
  jac = bunch.jac
  return (value=Vector(bunch.v[1, 1:6]), jacobian=Matrix(jac[1, :, :]), state=bunch.state[1])
end
