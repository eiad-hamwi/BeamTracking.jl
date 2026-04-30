@testset "KA Float32 widening + validator" begin
  using BeamTracking
  using BeamTracking: Coords, KernelCall, KernelChain
  using Beamlines
  using KernelAbstractions

  # Direct validator smoke: Float64 leaves get rewritten to Float32 when use_KA=true.
  b0 = Bunch(rand(Float32, 4, 6); jacobian=false, species=Species("electron"), p_over_q_ref=Float32(-1.0))
  coords = b0.coords
  kc = KernelChain((KernelCall(BeamTracking.exact_drift!, (1.0, 2.0, 3.0, 4.0)),))
  kc2 = validate_kernelchain(coords, kc; use_KA=true, groupsize=32)
  @test kc2.chain[1].args isa Tuple
  @test all(x -> x isa Float32, kc2.chain[1].args)

  # BeamlinesExt smoke: run a simple beamline with Float32 coordinates through KA (CPU backend).
  bl = Beamline(
    [
      Quadrupole(L=Float64(1.0), Kn1=Float64(0.13)),
      Drift(L=Float64(0.5)),
      Quadrupole(L=Float64(1.0), Kn1=Float64(-0.124)),
    ],
    species_ref=Species("proton"),
    p_over_q_ref=Float64(12.1),
  )
  b = Bunch(rand(Float32, 16, 6); jacobian=false, species=bl.species_ref, p_over_q_ref=Float32(bl.p_over_q_ref))
  track!(b, bl; use_KA=true, groupsize=64, use_explicit_SIMD=false)
  @test eltype(b.coords.v) === Float32
end

