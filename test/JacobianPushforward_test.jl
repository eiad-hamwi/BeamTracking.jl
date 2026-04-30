using Test,
  BeamTracking,
  GTPSA,
  StaticArrays,
  LinearAlgebra,
  ForwardDiff

import DifferentiationInterface as DI
using ADTypes: AutoGTPSA

using BeamTracking:
  Bunch,
  Coords,
  KernelCall,
  KernelChain,
  STATE_ALIVE,
  XI, PXI, YI, PYI, ZI, PZI,
  beamtracking_flow,
  AutoBeamTracking,
  beamtracking_jacobian_flat

const DJAC1 = Descriptor(6, 1)

function jac_coords(x0::AbstractVector{T}) where {T}
  v = reshape(copy(x0), 1, 6)
  state = fill(STATE_ALIVE, 1)
  jac = zeros(T, 1, 6, 6)
  BeamTracking.identity_jacobian!(jac)
  return Coords(state, v, nothing, nothing, (), jac)
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
  coords = jac_coords(x0)
  BeamTracking.launch!(coords, kcall_or_chain; use_KA=false, use_explicit_SIMD=false)
  return vec(coords.v[1,1:6]), Matrix(coords.jac[1,:,:])
end

function pushforward_result_mode(kcall_or_chain, x0; use_KA=false, use_explicit_SIMD=false, groupsize=nothing)
  coords = jac_coords(x0)
  BeamTracking.launch!(coords, kcall_or_chain; use_KA=use_KA, use_explicit_SIMD=use_explicit_SIMD, groupsize=groupsize)
  return vec(coords.v[1,1:6]), Matrix(coords.jac[1,:,:]), copy(coords.state)
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

    @test_throws ErrorException Coords(fill(STATE_ALIVE, 1), reshape(copy(x0), 1, 6), nothing, nothing, (), zeros(1, 5, 6))
    bunch_s = Bunch(v=reshape(copy(x0), 1, 6), spin=true)
    J_id = BeamTracking.identity_jacobian!(similar(bunch_s.v, 1, 6, 6))
    @test_throws ErrorException BeamTracking.launch!(
      Coords(bunch_s.state, bunch_s.v, bunch_s.q, bunch_s.weight, bunch_s.callbacks, J_id), k;
      use_KA=false, use_explicit_SIMD=false,
    )
    coords = jac_coords(x0)
    y_scalar, J_scalar, st_scalar = pushforward_result_mode(k, x0; use_KA=false, use_explicit_SIMD=false)
    y_simd, J_simd, st_simd = pushforward_result_mode(k, x0; use_KA=false, use_explicit_SIMD=true)
    y_ka, J_ka, st_ka = pushforward_result_mode(k, x0; use_KA=true, use_explicit_SIMD=false, groupsize=64)
    @test y_simd ≈ y_scalar
    @test J_simd ≈ J_scalar
    @test st_simd == st_scalar
    @test y_ka ≈ y_scalar
    @test J_ka ≈ J_scalar
    @test st_ka == st_scalar
    @test_throws ErrorException BeamTracking.launch!(coords, k; use_KA=true, use_explicit_SIMD=true)
    @test_throws ErrorException BeamTracking.launch!(coords, KernelCall(BeamTracking.isochronous_drift!, (0.1,)); use_KA=false, use_explicit_SIMD=false)
    @test_throws ErrorException BeamTracking.launch!(coords, KernelCall(BeamTracking.dkd_multipole!, (0.0, 0.0, true, beta_0, gamsqr_0, tilde_m, 0.0, ms, kn, ks, 0.29)); use_KA=false, use_explicit_SIMD=false)
  end

  @testset "BatchParam and Time compatibility" begin
    xbatch = [
      1.0e-3  2.0e-4 -7.0e-4 -3.0e-4  4.0e-3  1.0e-4;
      1.1e-3 -1.5e-4 -4.0e-4  2.5e-4 -3.0e-3 -2.0e-4;
      -8.0e-4 1.8e-4  5.0e-4 -1.4e-4  2.0e-3  8.0e-5;
      6.0e-4 -2.2e-4 -3.0e-4  1.1e-4 -1.0e-3  2.0e-4;
      -4.0e-4 1.0e-4  9.0e-4 -2.5e-4  3.0e-3 -1.0e-4;
      7.0e-4 -1.8e-4  2.0e-4  3.0e-4 -2.0e-3  1.5e-4
    ]
    state = fill(STATE_ALIVE, size(xbatch, 1))

    # BatchParam in kernel args should route per-particle values correctly in jacobian path.
    ds_batch = [0.11, 0.13, 0.17]
    k_batch = KernelCall(BeamTracking.exact_drift!, (beta_0, gamsqr_0, tilde_m, BatchParam(ds_batch)))
    jac_b = similar(xbatch, size(xbatch, 1), 6, 6)
    BeamTracking.identity_jacobian!(jac_b)
    coords_b = Coords(copy(state), copy(xbatch), nothing, nothing, (), jac_b)
    BeamTracking.launch!(coords_b, k_batch; use_KA=false, use_explicit_SIMD=false)
    jac_b_simd = similar(xbatch, size(xbatch, 1), 6, 6)
    BeamTracking.identity_jacobian!(jac_b_simd)
    coords_b_simd = Coords(copy(state), copy(xbatch), nothing, nothing, (), jac_b_simd)
    BeamTracking.launch!(coords_b_simd, k_batch; use_KA=false, use_explicit_SIMD=true)
    for i in 1:size(xbatch, 1)
      ds_i = ds_batch[mod1(i, length(ds_batch))]
      y_ref, J_ref = pushforward_result(KernelCall(BeamTracking.exact_drift!, (beta_0, gamsqr_0, tilde_m, ds_i)), vec(xbatch[i, :]))
      @test vec(coords_b.v[i, 1:6]) ≈ y_ref atol=1e-12 rtol=1e-12
      @test Matrix(coords_b.jac[i, :, :]) ≈ J_ref atol=1e-12 rtol=1e-12
      @test vec(coords_b_simd.v[i, 1:6]) ≈ y_ref atol=1e-12 rtol=1e-12
      @test Matrix(coords_b_simd.jac[i, :, :]) ≈ J_ref atol=1e-12 rtol=1e-12
    end

    # Time-dependent args should evaluate against per-particle time via RefState.
    t_ref = 0.23
    ref = BeamTracking.RefState(t_ref, BeamTracking.R_to_beta_gamma(species, p_over_q_ref))
    ds_time = 0.21 + 0.03 * sin(2 * Time())
    k_time = KernelChain((KernelCall(BeamTracking.exact_drift!, (beta_0, gamsqr_0, tilde_m, ds_time)),), ref)
    jac_t = similar(xbatch, size(xbatch, 1), 6, 6)
    BeamTracking.identity_jacobian!(jac_t)
    coords_t = Coords(copy(state), copy(xbatch), nothing, nothing, (), jac_t)
    BeamTracking.launch!(coords_t, k_time; use_KA=false, use_explicit_SIMD=false)
    jac_t_simd = similar(xbatch, size(xbatch, 1), 6, 6)
    BeamTracking.identity_jacobian!(jac_t_simd)
    coords_t_simd = Coords(copy(state), copy(xbatch), nothing, nothing, (), jac_t_simd)
    BeamTracking.launch!(coords_t_simd, k_time; use_KA=false, use_explicit_SIMD=true)
    for i in 1:size(xbatch, 1)
      t_i = BeamTracking.compute_time(xbatch[i, ZI], xbatch[i, PZI], ref)
      ds_i = 0.21 + 0.03 * sin(2 * t_i)
      y_ref, J_ref = pushforward_result(KernelCall(BeamTracking.exact_drift!, (beta_0, gamsqr_0, tilde_m, ds_i)), vec(xbatch[i, :]))
      @test vec(coords_t.v[i, 1:6]) ≈ y_ref atol=1e-12 rtol=1e-12
      @test Matrix(coords_t.jac[i, :, :]) ≈ J_ref atol=1e-12 rtol=1e-12
      @test vec(coords_t_simd.v[i, 1:6]) ≈ y_ref atol=1e-12 rtol=1e-12
      @test Matrix(coords_t_simd.jac[i, :, :]) ≈ J_ref atol=1e-12 rtol=1e-12
    end
  end

  @testset "track!(Bunch with jacobian)" begin
    k = KernelCall(BeamTracking.exact_drift!, (beta_0, gamsqr_0, tilde_m, 0.37))
    y_ref, J_ref = pushforward_result(k, x0)
    bunch = Bunch(v=reshape(copy(x0), 1, 6), species=species, p_over_q_ref=p_over_q_ref, jacobian=true)
    BeamTracking.track!(bunch, k; use_KA=false, use_explicit_SIMD=false)
    @test vec(bunch.v[1, 1:6]) ≈ y_ref atol=1e-11 rtol=1e-11
    @test Matrix(bunch.jac[1, :, :]) ≈ J_ref atol=1e-11 rtol=1e-11
  end

  @testset "DifferentiationInterface: analytic Jacobian + outer derivatives (ForwardDiff, GTPSA)" begin
    k = KernelCall(BeamTracking.exact_drift!, (beta_0, gamsqr_0, tilde_m, 0.37))
    flow = beamtracking_flow(k; species=species, p_over_q_ref=p_over_q_ref, use_KA=false, use_explicit_SIMD=false)
    y_ref, J_ref = gtpsa_reference(k, x0)
    y_di, J_di = DI.value_and_jacobian(flow, AutoBeamTracking(), x0)
    @test y_di ≈ y_ref atol=1e-11 rtol=1e-11
    @test J_di ≈ J_ref atol=1e-11 rtol=1e-11
    @test DI.jacobian(flow, AutoBeamTracking(), x0) ≈ J_ref atol=1e-11 rtol=1e-11

    # Outer AD touches only the Jacobian map (not beamtracking primal).
    jacflat = beamtracking_jacobian_flat(flow)
    H_from_di = DI.jacobian(jacflat, DI.AutoForwardDiff(), x0)
    d_outer = Descriptor(6, 3)
    H_from_gtpsa = DI.jacobian(jacflat, AutoGTPSA(; descriptor=d_outer), x0)
    @test H_from_gtpsa ≈ H_from_di atol=1e-12 rtol=1e-12

    map_value = x -> begin
      coords = Coords(fill(STATE_ALIVE, 1), reshape(copy(collect(x)), 1, 6), nothing, nothing, ())
      BeamTracking.launch!(coords, k; use_KA=false, use_explicit_SIMD=false)
      vec(coords.v[1, 1:6])
    end
    for row in 1:6
      H_ref_row = ForwardDiff.hessian(x -> map_value(x)[row], x0)
      H_row = [H_from_di[(col - 1) * 6 + row, j] for j in 1:6, col in 1:6]
      @test H_row ≈ H_ref_row atol=1e-10 rtol=1e-10
    end
  end
end
