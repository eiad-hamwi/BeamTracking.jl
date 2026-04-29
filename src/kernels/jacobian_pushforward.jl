struct Jet6{T}
  x::T
  dx::SVector{6,T}
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
@inline Base.:-(a::Jet6, b::Jet6) = Jet6(a.x - b.x, a.dx - b.dx)
@inline Base.:-(a::Jet6, b::Number) = Jet6(a.x - b, a.dx)
@inline Base.:-(a::Number, b::Jet6) = Jet6(a - b.x, -b.dx)
@inline Base.:-(a::Jet6) = Jet6(-a.x, -a.dx)
@inline Base.:*(a::Jet6, b::Jet6) = Jet6(a.x*b.x, a.x*b.dx + b.x*a.dx)
@inline Base.:*(a::Jet6, b::Number) = Jet6(a.x*b, a.dx*b)
@inline Base.:*(a::Number, b::Jet6) = Jet6(a*b.x, a*b.dx)
@inline Base.:/(a::Jet6, b::Jet6) = Jet6(a.x/b.x, (a.dx*b.x - a.x*b.dx)/(b.x*b.x))
@inline Base.:/(a::Jet6, b::Number) = Jet6(a.x/b, a.dx/b)
@inline Base.:/(a::Number, b::Jet6) = Jet6(a/b.x, -a*b.dx/(b.x*b.x))
@inline Base.sqrt(a::Jet6) = (s = sqrt(a.x); Jet6(s, a.dx/(2s)))
@inline Base.sin(a::Jet6) = Jet6(sin(a.x), cos(a.x)*a.dx)
@inline Base.cos(a::Jet6) = Jet6(cos(a.x), -sin(a.x)*a.dx)
@inline Base.tan(a::Jet6) = (t = tan(a.x); Jet6(t, (one(a.x) + t*t)*a.dx))
@inline Base.asin(a::Jet6) = Jet6(asin(a.x), a.dx / sqrt(one(a.x) - a.x*a.x))
@inline Base.sinh(a::Jet6) = Jet6(sinh(a.x), cosh(a.x)*a.dx)
@inline Base.cosh(a::Jet6) = Jet6(cosh(a.x), sinh(a.x)*a.dx)
@inline Base.atan(y::Jet6, x::Jet6) = (d = x.x*x.x + y.x*y.x; Jet6(atan(y.x, x.x), (x.x*y.dx - y.x*x.dx)/d))
@inline Base.abs(a::Jet6) = Jet6(abs(a.x), sign(a.x)*a.dx)
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

@inline _lane_eps(x::T) where {T<:Number} = eps(T)
@inline _lane_eps(x::SIMD.Vec{N,T}) where {N,T} = SIMD.Vec{N,T}(eps(T))

@inline function sincu(a::Jet6)
  s = sincu(a.x)
  thr = _lane_eps(a.x)^(1/3)
  ds = vifelse(abs(a.x) < thr, zero(a.x), (cos(a.x) - s)/a.x)
  return Jet6(s, ds*a.dx)
end

@inline function sinhcu(a::Jet6)
  s = sinhcu(a.x)
  thr = _lane_eps(a.x)^(1/3)
  ds = vifelse(abs(a.x) < thr, zero(a.x), (cosh(a.x) - s)/a.x)
  return Jet6(s, ds*a.dx)
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

@inline function exact_drift_with_jac!(i, coords::Coords, jac, beta_0, gamsqr_0, tilde_m, L)
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
  left_compose_jac6_masked!(jac, i, A, alive)
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

@inline function rotation_with_jac!(i, coords::Coords, jac, q_inv, z_0)
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
  T = promote_type(typeof(Ps), typeof(w11), typeof(w12), typeof(w13), typeof(w21), typeof(w22), typeof(w23))
  A = smat6_from_entries(T) do r, c
    if r == c && (r == ZI || r == PZI)
      one(T)
    elseif r == XI && c == XI
      w11
    elseif r == XI && c == YI
      w12
    elseif r == YI && c == XI
      w21
    elseif r == YI && c == YI
      w22
    elseif r == PXI && c == PXI
      w11 - w13*px0/Ps
    elseif r == PXI && c == PYI
      w12 - w13*py0/Ps
    elseif r == PXI && c == PZI
      w13*P/Ps
    elseif r == PYI && c == PXI
      w21 - w23*px0/Ps
    elseif r == PYI && c == PYI
      w22 - w23*py0/Ps
    elseif r == PYI && c == PZI
      w23*P/Ps
    else
      zero(T)
    end
  end
  left_compose_jac6_masked!(jac, i, A, alive)
  return nothing
end

@inline function linear_bend_fringe_with_jac!(i, coords::Coords, jac, a, tilde_m, Ksol, Kn0, e, sign)
  alive = coords.state[i] == STATE_ALIVE
  v = coords.v
  f = Kn0*tan(e)
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
  left_compose_jac6_masked!(jac, i, A, alive)
  return nothing
end

@inline function multipole_kick_map6(x, ms, knl, ksl, excluding)
  X, PX, Y, PY, Z, PZ = x
  bx, by = normalized_field(ms, knl, ksl, X, Y, excluding)
  return SVector(X, PX - by, Y, PY + bx, Z, PZ)
end

@inline function multipole_kick_with_jac!(i, coords::Coords, jac, ms, knl, ksl, excluding)
  alive = coords.state[i] == STATE_ALIVE
  x1, A = map6_value_and_jac(multipole_kick_map6, coord_svec(coords, i), ms, knl, ksl, excluding)
  masked_store_coord_svec!(coords, i, x1, alive)
  left_compose_jac6_masked!(jac, i, A, alive)
  return nothing
end

@inline function multipole_and_spin_kick_with_jac!(i, coords::Coords, jac, mm, kn, ks, a, tilde_m, L)
  return multipole_kick_with_jac!(i, coords, jac, mm, kn .* L, ks .* L, 0)
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

@inline function quadrupole_kick_with_jac!(i, coords::Coords, jac, beta_0, gamsqr_0, tilde_m, s)
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
  left_compose_jac6_masked!(jac, i, A, alive)
  return nothing
end

@inline function quadrupole_matrix_map6(x, k1, s)
  X, PX, Y, PY, Z, PZ = x
  focus = k1 >= 0
  P = one(PZ) + PZ
  xp = PX / P
  yp = PY / P
  sqrtks = sqrt(abs(k1 / P)) * s
  cosine = cos(sqrtks)
  coshine = cosh(sqrtks)
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

@inline function quadrupole_matrix_with_jac!(i, coords::Coords, jac, k1, s)
  alive = coords.state[i] == STATE_ALIVE
  x1, A = map6_value_and_jac(quadrupole_matrix_map6, coord_svec(coords, i), k1, s)
  masked_store_coord_svec!(coords, i, x1, alive)
  left_compose_jac6_masked!(jac, i, A, alive)
  return nothing
end

@inline function exact_bend_cond(x, theta, g, Kn0, L)
  X, PX, Y, PY, Z, PZ = x
  P = one(PZ) + PZ
  pt = sqrt(P*P - PY*PY)
  arg = PX / pt
  phi1 = theta + asin(arg)
  gp = Kn0 / pt
  h = one(X) + g*X
  cplus = cos(phi1)
  splus = sin(phi1)
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
  phi1 = theta + asin(arg)
  gp = Kn0 / pt
  h = one(X) + g*X
  cplus = cos(phi1)
  splus = sin(phi1)
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
  Lcv = -sgn*(L*sinc_theta + X*sin(theta))
  negative_Lcv = -Lcv
  thetap = 2*(phi1 - sgn*atan(xi, negative_Lcv))
  Lp = sgn*sqrt(Lcv*Lcv + xi*xi)/sincu(thetap/2)
  Xn = X*cos(theta) - L*L*g*cosc_theta + xi
  PXn = pt*sin(phi1 - thetap)
  Yn = Y + PY*Lp/pt
  Zn = Z - P*Lp/pt + L*P/sqrt(tilde_m*tilde_m + P*P)/beta_0
  return SVector(Xn, PXn, Yn, PY, Zn, PZ)
end

@inline function exact_bend_with_jac!(i, coords::Coords, jac, theta, g, Kn0, tilde_m, beta_0, L)
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
  left_compose_jac6_masked!(jac, i, A, alive)
  return nothing
end

@inline function dkd_multipole_with_jac!(i, coords::Coords, jac, q, mc2, radiation_damping, beta_0, gamsqr_0, tilde_m, a, mm, kn, ks, L)
  exact_drift_with_jac!(i, coords, jac, beta_0, gamsqr_0, tilde_m, L / 2)
  multipole_and_spin_kick_with_jac!(i, coords, jac, mm, kn, ks, a, tilde_m, L)
  exact_drift_with_jac!(i, coords, jac, beta_0, gamsqr_0, tilde_m, L / 2)
  return nothing
end

@inline function bkb_multipole_with_jac!(i, coords::Coords, jac, q, mc2, radiation_damping, tilde_m, beta_0, a, g, w, w_inv, k0, mm, kn, ks, L)
  knl = kn .* L ./ 2
  ksl = ks .* L ./ 2
  rotation_with_jac!(i, coords, jac, w, 0)
  multipole_kick_with_jac!(i, coords, jac, mm, knl, ksl, 1)
  exact_bend_with_jac!(i, coords, jac, g*L, g, k0, tilde_m, beta_0, L)
  multipole_kick_with_jac!(i, coords, jac, mm, knl, ksl, 1)
  rotation_with_jac!(i, coords, jac, w_inv, 0)
  return nothing
end

@inline function mkm_quadrupole_with_jac!(i, coords::Coords, jac, q, mc2, radiation_damping, beta_0, gamsqr_0, tilde_m, a, w, w_inv, k1, mm, kn, ks, L)
  alive = coords.state[i] == STATE_ALIVE
  x0 = coord_svec(coords, i)
  P = one(x0[PZI]) + x0[PZI]
  Ps2 = P*P - x0[PXI]*x0[PXI] - x0[PYI]*x0[PYI]
  coords.state[i] = vifelse((!(Ps2 > zero(Ps2))) & alive, STATE_LOST, coords.state[i])
  knl = kn .* L ./ 2
  ksl = ks .* L ./ 2
  multipole_kick_with_jac!(i, coords, jac, mm, knl, ksl, 2)
  quadrupole_kick_with_jac!(i, coords, jac, beta_0, gamsqr_0, tilde_m, L / 2)
  rotation_with_jac!(i, coords, jac, w, 0)
  quadrupole_matrix_with_jac!(i, coords, jac, k1, L)
  rotation_with_jac!(i, coords, jac, w_inv, 0)
  quadrupole_kick_with_jac!(i, coords, jac, beta_0, gamsqr_0, tilde_m, L / 2)
  multipole_kick_with_jac!(i, coords, jac, mm, knl, ksl, 2)
  return nothing
end

@inline function order_two_integrator_with_jac!(i, coords::Coords, jac, ker, params, photon_params, ds_step, num_steps, edge_params, ::Val{fringe_in}, ::Val{fringe_out}, L) where {fringe_in,fringe_out}
  if !isnothing(edge_params) && fringe_in
    a, tilde_m, Ksol, Kn0, e1, e2 = edge_params
    linear_bend_fringe_with_jac!(i, coords, jac, a, tilde_m, Ksol, Kn0, e1, 1)
  end
  for step in 1:num_steps
    ker(i, coords, jac, params..., ds_step)
  end
  if !isnothing(edge_params) && fringe_out
    a, tilde_m, Ksol, Kn0, e1, e2 = edge_params
    linear_bend_fringe_with_jac!(i, coords, jac, a, tilde_m, Ksol, Kn0, e2, -1)
  end
  return nothing
end

@inline function order_four_integrator_with_jac!(i, coords::Coords, jac, ker, params, photon_params, ds_step, num_steps, edge_params, ::Val{fringe_in}, ::Val{fringe_out}, L) where {fringe_in,fringe_out}
  w0 = -1.7024143839193153215916254339390434324741363525390625*ds_step
  w1 =  1.3512071919596577718181151794851757586002349853515625*ds_step
  if !isnothing(edge_params) && fringe_in
    a, tilde_m, Ksol, Kn0, e1, e2 = edge_params
    linear_bend_fringe_with_jac!(i, coords, jac, a, tilde_m, Ksol, Kn0, e1, 1)
  end
  for step in 1:num_steps
    ker(i, coords, jac, params..., w1)
    ker(i, coords, jac, params..., w0)
    ker(i, coords, jac, params..., w1)
  end
  if !isnothing(edge_params) && fringe_out
    a, tilde_m, Ksol, Kn0, e1, e2 = edge_params
    linear_bend_fringe_with_jac!(i, coords, jac, a, tilde_m, Ksol, Kn0, e2, -1)
  end
  return nothing
end

@inline function order_six_integrator_with_jac!(i, coords::Coords, jac, ker, params, photon_params, ds_step, num_steps, edge_params, ::Val{fringe_in}, ::Val{fringe_out}, L) where {fringe_in,fringe_out}
  w0 =  1.315186320683911169737712043570355*ds_step
  w1 = -1.17767998417887100694641568096432*ds_step
  w2 =  0.235573213359358133684793182978535*ds_step
  w3 =  0.784513610477557263819497633866351*ds_step
  if !isnothing(edge_params) && fringe_in
    a, tilde_m, Ksol, Kn0, e1, e2 = edge_params
    linear_bend_fringe_with_jac!(i, coords, jac, a, tilde_m, Ksol, Kn0, e1, 1)
  end
  for step in 1:num_steps
    ker(i, coords, jac, params..., w3)
    ker(i, coords, jac, params..., w2)
    ker(i, coords, jac, params..., w1)
    ker(i, coords, jac, params..., w0)
    ker(i, coords, jac, params..., w1)
    ker(i, coords, jac, params..., w2)
    ker(i, coords, jac, params..., w3)
  end
  if !isnothing(edge_params) && fringe_out
    a, tilde_m, Ksol, Kn0, e1, e2 = edge_params
    linear_bend_fringe_with_jac!(i, coords, jac, a, tilde_m, Ksol, Kn0, e2, -1)
  end
  return nothing
end

@inline function order_eight_integrator_with_jac!(i, coords::Coords, jac, ker, params, photon_params, ds_step, num_steps, edge_params, ::Val{fringe_in}, ::Val{fringe_out}, L) where {fringe_in,fringe_out}
  w0 =  1.7084530707869978*ds_step
  w1 =  0.102799849391985*ds_step
  w2 = -1.96061023297549*ds_step
  w3 =  1.93813913762276*ds_step
  w4 = -0.158240635368243*ds_step
  w5 = -1.44485223686048*ds_step
  w6 =  0.253693336566229*ds_step
  w7 =  0.914844246229740*ds_step
  if !isnothing(edge_params) && fringe_in
    a, tilde_m, Ksol, Kn0, e1, e2 = edge_params
    linear_bend_fringe_with_jac!(i, coords, jac, a, tilde_m, Ksol, Kn0, e1, 1)
  end
  for step in 1:num_steps
    ker(i, coords, jac, params..., w7)
    ker(i, coords, jac, params..., w6)
    ker(i, coords, jac, params..., w5)
    ker(i, coords, jac, params..., w4)
    ker(i, coords, jac, params..., w3)
    ker(i, coords, jac, params..., w2)
    ker(i, coords, jac, params..., w1)
    ker(i, coords, jac, params..., w0)
    ker(i, coords, jac, params..., w1)
    ker(i, coords, jac, params..., w2)
    ker(i, coords, jac, params..., w3)
    ker(i, coords, jac, params..., w4)
    ker(i, coords, jac, params..., w5)
    ker(i, coords, jac, params..., w6)
    ker(i, coords, jac, params..., w7)
  end
  if !isnothing(edge_params) && fringe_out
    a, tilde_m, Ksol, Kn0, e1, e2 = edge_params
    linear_bend_fringe_with_jac!(i, coords, jac, a, tilde_m, Ksol, Kn0, e2, -1)
  end
  return nothing
end

_generic_kernel_with_jac!(i, coords, jac, kc) = __generic_kernel_with_jac!(i, coords, jac, kc.chain, kc.ref)

@kernel function generic_kernel_with_jac!(coords::Coords, jac, @Const(kc::KernelChain))
  i = @index(Global, Linear)
  @inline _generic_kernel_with_jac!(i, coords, jac, kc)
end

@unroll function __generic_kernel_with_jac!(i, coords::Coords, jac, chain, ref)
  @unroll for kcall in chain
    bargs = process_batch_args(i, kcall.args)
    args = process_time_args(i, coords, bargs, ref)
    (kcall.kernel)(i, coords, jac, args...)
  end
  return nothing
end

@inline function check_jacobian_sidecar(coords::Coords, jac)
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

@inline function preflight_jacobian_tracking(coords::Coords, jac, kc::KernelChain; use_KA::Bool, use_explicit_SIMD::Bool)
  check_jacobian_sidecar(coords, jac)
  use_KA && use_explicit_SIMD && error("Cannot use both KernelAbstractions (KA) and explicit SIMD")
  for kcall in kc.chain
    _preflight_jacobian_kcall(kcall)
  end
  if use_KA
    get_backend(jac) == get_backend(coords.v) || error("Jacobian sidecar backend must match coords.v backend")
  end
  return nothing
end

@inline function launch_with_jac!(
  coords::Coords,
  jac,
  kc::KernelChain;
  groupsize::Union{Nothing,Integer}=nothing,
  multithread_threshold::Integer=Threads.nthreads() > 1 ? 1750*Threads.nthreads() : typemax(Int),
  use_KA::Bool=!(get_backend(coords.v) isa CPU && isnothing(groupsize)),
  use_explicit_SIMD::Bool=!use_KA
)
  preflight_jacobian_tracking(coords, jac, kc; use_KA=use_KA, use_explicit_SIMD=use_explicit_SIMD)
  v = coords.v
  N_particle = size(coords.v, 1)
  if !use_KA
    if use_explicit_SIMD && v isa SIMD.FastContiguousArray && eltype(v) <: SIMD.ScalarTypes && pick_vector_width(eltype(v)) > 1
      simd_lane_width = pick_vector_width(eltype(v))
      lane = SIMD.VecRange{Int(simd_lane_width)}(0)
      rmn = rem(N_particle, simd_lane_width)
      N_SIMD = N_particle - rmn
      if N_particle >= multithread_threshold
        Threads.@threads for i in 1:simd_lane_width:N_SIMD
          @assert last(i) <= N_particle "Out of bounds!"
          _generic_kernel_with_jac!(lane+i, coords, jac, kc)
        end
      else
        for i in 1:simd_lane_width:N_SIMD
          @assert last(i) <= N_particle "Out of bounds!"
          _generic_kernel_with_jac!(lane+i, coords, jac, kc)
        end
      end
      for i in N_SIMD+1:N_particle
        @assert last(i) <= N_particle "Out of bounds!"
        _generic_kernel_with_jac!(i, coords, jac, kc)
      end
    elseif N_particle >= multithread_threshold
      Threads.@threads for i in 1:N_particle
        _generic_kernel_with_jac!(i, coords, jac, kc)
      end
    else
      @simd for i in 1:N_particle
        _generic_kernel_with_jac!(i, coords, jac, kc)
      end
    end
  else
    backend = get_backend(v)
    if isnothing(groupsize)
      kernel! = generic_kernel_with_jac!(backend)
    else
      kernel! = generic_kernel_with_jac!(backend, groupsize)
    end
    kernel!(coords, jac, kc; ndrange=N_particle)
    KernelAbstractions.synchronize(backend)
  end
  return nothing
end

@inline launch_with_jac!(coords::Coords, jac, kcall::KernelCall; kwargs...) =
  launch_with_jac!(coords, jac, KernelChain((kcall,)); kwargs...)

@inline blank_kernel_with_jac!(args...) = nothing

@inline _jacobian_kernel(::typeof(blank_kernel!)) = blank_kernel_with_jac!
@inline _jacobian_kernel(::typeof(exact_drift!)) = exact_drift_with_jac!
@inline _jacobian_kernel(::typeof(rotation!)) = rotation_with_jac!
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

function track_with_jac!(coords::Coords, jac, kcall_or_chain; kwargs...)
  launch_with_jac!(coords, jac, jacobianize(kcall_or_chain); kwargs...)
  return coords, jac
end

function track_with_jac!(bunch::Bunch, jac, kcall_or_chain; kwargs...)
  track_with_jac!(bunch.coords, jac, kcall_or_chain; kwargs...)
  return bunch, jac
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
  bunch = Bunch(v=v, species=species, p_over_q_ref=p_over_q_ref, t_ref=t_ref)
  jac = identity_jacobian!(allocate_coordinate_jacobian(bunch))
  track_with_jac!(bunch, jac, kcall_or_chain; kwargs...)
  return (value=Vector(bunch.v[1,1:6]), jacobian=Matrix(jac[1,:,:]), state=bunch.state[1])
end
