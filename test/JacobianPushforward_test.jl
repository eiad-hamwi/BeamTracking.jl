using Test,
  BeamTracking,
  GTPSA,
  StaticArrays,
  LinearAlgebra

using BeamTracking:
  Bunch,
  Coords,
  KernelCall,
  KernelChain,
  STATE_ALIVE,
  XI, PXI, YI, PYI, ZI, PZI

const DJAC1 = Descriptor(6, 1)

function jac_real_coords(x0::AbstractVector{T}) where {T}
  v = reshape(copy(x0), 1, 6)
  state = fill(STATE_ALIVE, 1)
  return Coords(state, v, nothing, nothing, ())
end

function jac_coords(x0::AbstractVector{T}) where {T}
  coords = jac_real_coords(x0)
  jac = zeros(T, 1, 6, 6)
  BeamTracking.identity_jacobian!(jac)
  return coords, jac
end

function gtpsa_coords(x0::AbstractVector{T}) where {T}
  v = Matrix{TPS64{DJAC1}}(undef, 1, 6)
  vars1 = vars(DJAC1)
  for j in 1:6
    v[1,j] = x0[j] + vars1[j]
  end
  state = fill(STATE_ALIVE, 1)
  return Coords(state, v, nothing, nothing, ())
end

scalarize_row(v) = Float64[scalar(v[1,j]) for j in 1:6]

function gtpsa_reference(kcall_or_chain, x0)
  coords = gtpsa_coords(x0)
  BeamTracking.launch!(coords, kcall_or_chain; use_KA=false, use_explicit_SIMD=false)
  return scalarize_row(coords.v), Matrix{Float64}(GTPSA.jacobian(coords.v)[1:6, 1:6])
end

function pushforward_result(kcall_or_chain, x0)
  coords, J = jac_coords(x0)
  BeamTracking.track_with_jac!(coords, J, kcall_or_chain; use_KA=false, use_explicit_SIMD=false)
  return vec(coords.v[1,1:6]), Matrix(J[1,:,:])
end

function pushforward_result_mode(kcall_or_chain, x0; use_KA=false, use_explicit_SIMD=false, groupsize=nothing)
  coords, J = jac_coords(x0)
  BeamTracking.track_with_jac!(coords, J, kcall_or_chain; use_KA=use_KA, use_explicit_SIMD=use_explicit_SIMD, groupsize=groupsize)
  return vec(coords.v[1,1:6]), Matrix(J[1,:,:]), copy(coords.state)
end

function compare_pushforward(kcall_or_chain, x0; atol=1e-10, rtol=1e-10)
  y_ref, J_ref = gtpsa_reference(kcall_or_chain, x0)
  y_new, J_new = pushforward_result(kcall_or_chain, x0)
  @test y_new ≈ y_ref atol=atol rtol=rtol
  @test J_new ≈ J_ref atol=atol rtol=rtol
end

function symplectic_form6()
  Ω = zeros(6, 6)
  for (a, b) in ((XI, PXI), (YI, PYI), (ZI, PZI))
    Ω[a,b] = 1
    Ω[b,a] = -1
  end
  return Ω
end

function compare_symplectic(kcall_or_chain, x0; atol=1e-10, rtol=1e-10)
  _y, J = pushforward_result(kcall_or_chain, x0)
  Ω = symplectic_form6()
  @test J' * Ω * J ≈ Ω atol=atol rtol=rtol
end

@testset "Coordinate Jacobian pushforward" begin
  species = Species("electron")
  p_over_q_ref = -6.0e7
  tilde_m, gamsqr_0, beta_0 = BeamTracking.drift_params(species, p_over_q_ref)
  x0 = [1e-3, 2e-4, -7e-4, -3e-4, 4e-3, 1e-4]
  qI = SA[1.0, 0.0, 0.0, 0.0]
  ms = SA[1, 2, 3]
  kn = SA[0.015, -0.03, 0.02]
  ks = SA[-0.01, 0.02, -0.015]

  @testset "primitive kernels" begin
    compare_pushforward(KernelCall(BeamTracking.exact_drift!, (beta_0, gamsqr_0, tilde_m, 0.37)), x0; atol=1e-11, rtol=1e-11)
    compare_pushforward(KernelCall(BeamTracking.rotation!, (SA[cos(0.02), 0.0, sin(0.02), 0.0], 0.0)), x0)
    compare_pushforward(KernelCall(BeamTracking.linear_bend_fringe!, (0.0, tilde_m, 0.0, 0.02, 0.12, 1)), x0)
    compare_pushforward(KernelCall(BeamTracking.multipole_kick!, (ms, kn .* 0.21, ks .* 0.21, -1)), x0)
    compare_pushforward(KernelCall(BeamTracking.quadrupole_kick!, (beta_0, gamsqr_0, tilde_m, 0.11)), x0)
    compare_pushforward(KernelCall(BeamTracking.quadrupole_matrix!, (0.18, 0.23)), x0)
    compare_pushforward(KernelCall(BeamTracking.exact_bend!, (0.012, 0.04, 0.04, tilde_m, beta_0, 0.30)), x0; atol=1e-9, rtol=1e-9)
  end

  @testset "composed kernels" begin
    compare_pushforward(KernelCall(BeamTracking.dkd_multipole!, (0.0, 0.0, false, beta_0, gamsqr_0, tilde_m, 0.0, ms, kn, ks, 0.29)), x0; atol=1e-10, rtol=1e-10)
    compare_pushforward(KernelCall(BeamTracking.bkb_multipole!, (0.0, 0.0, false, tilde_m, beta_0, 0.0, 0.04, qI, qI, 0.04, ms, kn, ks, 0.20)), x0; atol=1e-9, rtol=1e-9)
    compare_pushforward(KernelCall(BeamTracking.mkm_quadrupole!, (0.0, 0.0, false, beta_0, gamsqr_0, tilde_m, 0.0, qI, qI, 0.18, ms, kn, ks, 0.22)), x0; atol=1e-9, rtol=1e-9)
  end

  @testset "Yoshida drivers" begin
    params = (0.0, 0.0, false, beta_0, gamsqr_0, tilde_m, 0.0, ms, kn, ks)
    k2 = KernelCall(BeamTracking.order_two_integrator!, (BeamTracking.dkd_multipole!, params, nothing, 0.07, 3, nothing, Val(false), Val(false), 0.21))
    k4 = KernelCall(BeamTracking.order_four_integrator!, (BeamTracking.dkd_multipole!, params, nothing, 0.07, 2, nothing, Val(false), Val(false), 0.14))
    compare_pushforward(k2, x0; atol=1e-10, rtol=1e-10)
    compare_pushforward(k4, x0; atol=1e-9, rtol=1e-9)
  end

  @testset "symplecticity" begin
    compare_symplectic(KernelCall(BeamTracking.exact_drift!, (beta_0, gamsqr_0, tilde_m, 0.37)), x0; atol=1e-10, rtol=1e-10)
    compare_symplectic(KernelCall(BeamTracking.dkd_multipole!, (0.0, 0.0, false, beta_0, gamsqr_0, tilde_m, 0.0, ms, kn, ks, 0.29)), x0; atol=1e-9, rtol=1e-9)
  end

  @testset "public convenience API and failures" begin
    k = KernelCall(BeamTracking.exact_drift!, (beta_0, gamsqr_0, tilde_m, 0.37))
    result = BeamTracking.value_and_jacobian(x0, k; species=species, p_over_q_ref=p_over_q_ref, use_KA=false, use_explicit_SIMD=false)
    y_ref, J_ref = gtpsa_reference(k, x0)
    @test result.value ≈ y_ref
    @test result.jacobian ≈ J_ref
    @test result.state == STATE_ALIVE

    coords, J = jac_coords(x0)
    @test_throws ErrorException BeamTracking.track_with_jac!(coords, zeros(1, 5, 6), k; use_KA=false, use_explicit_SIMD=false)
    @test_throws ErrorException BeamTracking.track_with_jac!(Bunch(v=reshape(copy(x0), 1, 6), spin=true).coords, J, k; use_KA=false, use_explicit_SIMD=false)
    y_scalar, J_scalar, st_scalar = pushforward_result_mode(k, x0; use_KA=false, use_explicit_SIMD=false)
    y_simd, J_simd, st_simd = pushforward_result_mode(k, x0; use_KA=false, use_explicit_SIMD=true)
    y_ka, J_ka, st_ka = pushforward_result_mode(k, x0; use_KA=true, use_explicit_SIMD=false, groupsize=64)
    @test y_simd ≈ y_scalar
    @test J_simd ≈ J_scalar
    @test st_simd == st_scalar
    @test y_ka ≈ y_scalar
    @test J_ka ≈ J_scalar
    @test st_ka == st_scalar
    @test_throws ErrorException BeamTracking.track_with_jac!(coords, J, k; use_KA=true, use_explicit_SIMD=true)
    @test_throws ErrorException BeamTracking.track_with_jac!(coords, J, KernelCall(BeamTracking.isochronous_drift!, (0.1,)); use_KA=false, use_explicit_SIMD=false)
    @test_throws ErrorException BeamTracking.track_with_jac!(coords, J, KernelCall(BeamTracking.dkd_multipole!, (0.0, 0.0, true, beta_0, gamsqr_0, tilde_m, 0.0, ms, kn, ks, 0.29)); use_KA=false, use_explicit_SIMD=false)
  end
end
