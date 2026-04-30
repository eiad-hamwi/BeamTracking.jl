module BeamTrackingBeamlinesExt
using Beamlines, BeamTracking, GTPSA, StaticArrays, KernelAbstractions, AtomicAndPhysicalConstants, LinearAlgebra
using Beamlines: isactive, deval, unsafe_getparams, isnullspecies
using BeamTracking: R_to_E, R_to_beta_gamma, R_to_gamma, R_to_pc, R_to_v,
                    beta_gamma_to_v, E_to_R, E_to_v,
                    @makekernel, Coords, KernelCall, KernelChain, push, TimeDependentParam, RefState,
                    launch!, AbstractYoshida, rot_quaternion, inv_rot_quaternion, atan2,
                    get_N_particle, mean_and_cov, ibs_integrals

import BeamTracking: track!

@inline _element_name(ele::LineElement) = string(getproperty(ele, :name))

function track!(
  bunch::Bunch,
  ele::LineElement;
  scalar_params::Bool=false,
  ramp_particle_energy_without_rf::Bool=false,
  kwargs...
)
  coords = bunch.coords
  if bunch.jac === nothing
    @noinline _track!(coords, bunch, ele, ele.tracking_method, scalar_params, ramp_particle_energy_without_rf; kwargs...)
  else
    local tm = ele.tracking_method
    local ele_name = _element_name(ele)
    try
      @noinline _track!(coords, bunch, ele, tm, scalar_params, ramp_particle_energy_without_rf; kwargs...)
    catch err
      msg = sprint(showerror, err)
      error("Beamline Jacobian tracking failed at element \"$ele_name\" with tracking_method $(typeof(tm)): $msg")
    end
  end
  return bunch
end

function track!(
  bunch::Bunch,
  bl::Beamline;
  scalar_params::Bool=false,
  ramp_particle_energy_without_rf::Bool=false,
  kwargs...
)
  if length(bl.line) == 0
    return bunch
  end
  check_bl_bunch!(bl, bunch)

  if bunch.jac === nothing
    for ele in bl.line
      track!(bunch, ele; scalar_params, ramp_particle_energy_without_rf, kwargs...)
    end
  else
    for (i, ele) in pairs(bl.line)
      local ele_name = _element_name(ele)
      try
        track!(bunch, ele; scalar_params, ramp_particle_energy_without_rf, kwargs...)
      catch err
        msg = sprint(showerror, err)
        error("Beamline Jacobian tracking failed at element index $i (\"$ele_name\"): $msg")
      end
    end
  end

  return bunch
end

include("utils_bl.jl")
include("unpack_bl.jl")
include("scibmadstandard_bl.jl")
include("exact_bl.jl")
include("yoshida_bl.jl")
include("sagan_cavity_bl.jl")
include("general_bl.jl")

end
