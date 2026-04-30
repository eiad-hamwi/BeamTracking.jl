const btbl = Base.get_extension(BeamTracking, :BeamTrackingBeamlinesExt)
using ForwardDiff

function beamline_value(bl, x0; kwargs...)
  bunch = Bunch(reshape(copy(x0), 1, 6), species=Species("electron"))
  track!(bunch, bl; kwargs...)
  return vec(bunch.coords.v[1, 1:6]), copy(bunch.coords.state)
end

function beamline_value_and_jac(bl, x0; kwargs...)
  bunch = Bunch(reshape(copy(x0), 1, 6), species=Species("electron"), jacobian=true)
  track!(bunch, bl; kwargs...)
  jac = bunch.jac
  return vec(bunch.coords.v[1, 1:6]), Matrix(jac[1, :, :]), copy(bunch.coords.state)
end

function beamline_batch_value_and_jac(bl, x; kwargs...)
  bunch = Bunch(copy(x), species=Species("electron"), jacobian=true)
  track!(bunch, bl; kwargs...)
  jac = bunch.jac
  return copy(bunch.coords.v), copy(jac), copy(bunch.coords.state)
end

@testset "Beamlines Jacobian pushforward" begin
  x0 = [1e-3, 2e-4, -7e-4, -3e-4, 4e-3, 1e-4]

  d = Drift(L=0.37)
  sx = Sextupole(Kn2=0.08, L=0.29, tracking_method=DriftKick(order=4))
  qmk = Quadrupole(Kn1=0.23, L=0.18, tracking_method=MatrixKick(order=4))
  dbk = Drift(L=0.22, Kn0=0.03, tracking_method=BendKick(order=4))

  beamlines = (
    Beamline([d], species_ref=Species("electron"), E_ref=18e9),
    Beamline([sx], species_ref=Species("electron"), E_ref=18e9),
    Beamline([qmk], species_ref=Species("electron"), E_ref=18e9),
    Beamline([dbk], species_ref=Species("electron"), E_ref=18e9),
  )

  for bl in beamlines
    y_track, st_track = beamline_value(bl, x0; use_KA=false, use_explicit_SIMD=false)
    y_jac, J_jac, st_jac = beamline_value_and_jac(bl, x0; use_KA=false, use_explicit_SIMD=false)
    @test y_jac ≈ y_track atol=1e-11 rtol=1e-11
    @test st_jac == st_track

    J_fd = ForwardDiff.jacobian(x -> first(beamline_value(bl, x; use_KA=false, use_explicit_SIMD=false)), x0)
    @test J_jac ≈ J_fd atol=1e-8 rtol=1e-8
  end

  xbatch = [
    1.0e-3  2.0e-4 -7.0e-4 -3.0e-4  4.0e-3  1.0e-4;
    1.1e-3 -1.5e-4 -4.0e-4  2.5e-4 -3.0e-3 -2.0e-4;
    -8.0e-4 1.8e-4  5.0e-4 -1.4e-4  2.0e-3  8.0e-5;
    6.0e-4 -2.2e-4 -3.0e-4  1.1e-4 -1.0e-3  2.0e-4;
    -4.0e-4 1.0e-4  9.0e-4 -2.5e-4  3.0e-3 -1.0e-4;
    7.0e-4 -1.8e-4  2.0e-4  3.0e-4 -2.0e-3  1.5e-4;
    -9.0e-4 2.2e-4 -1.0e-4  2.0e-4  1.0e-3 -9.0e-5;
    4.0e-4 -1.0e-4  8.0e-4 -3.0e-4  2.5e-3  7.0e-5;
  ]
  bl_simd = Beamline([deepcopy(sx)], species_ref=Species("electron"), E_ref=18e9)
  y_scalar, J_scalar, st_scalar = beamline_batch_value_and_jac(bl_simd, xbatch; use_KA=false, use_explicit_SIMD=false)
  y_simd, J_simd, st_simd = beamline_batch_value_and_jac(bl_simd, xbatch; use_KA=false, use_explicit_SIMD=true)
  @test y_simd ≈ y_scalar atol=1e-12 rtol=1e-12
  @test J_simd ≈ J_scalar atol=1e-12 rtol=1e-12
  @test st_simd == st_scalar

  # Unsupported paths should error clearly.
  spin_bunch = Bunch(v=reshape(copy(x0), 1, 6), q=[1.0 0.0 0.0 0.0], species=Species("electron"), jacobian=true)
  @test_throws ErrorException track!(spin_bunch, beamlines[1]; use_KA=false, use_explicit_SIMD=false)

  rf_bl = Beamline([RFCavity(L=1e-2, voltage=1e6, rf_frequency=1e6)], species_ref=Species("electron"), E_ref=18e9)
  @test_throws ErrorException beamline_value_and_jac(rf_bl, x0; use_KA=false, use_explicit_SIMD=false)

  rad_ele = Sextupole(Kn2=0.08, L=0.29, tracking_method=DriftKick(order=4, radiation_fluctuations_on=true))
  rad_bl = Beamline([rad_ele], species_ref=Species("electron"), E_ref=18e9)
  @test_throws ErrorException beamline_value_and_jac(rad_bl, x0; use_KA=false, use_explicit_SIMD=false)

  sagan_ele = RFCavity(L=0.3, voltage=1e6, rf_frequency=1e6, tracking_method=SaganCavity())
  sagan_bl = Beamline([sagan_ele], species_ref=Species("electron"), E_ref=18e9)
  @test_throws ErrorException beamline_value_and_jac(sagan_bl, x0; use_KA=false, use_explicit_SIMD=false)
end
