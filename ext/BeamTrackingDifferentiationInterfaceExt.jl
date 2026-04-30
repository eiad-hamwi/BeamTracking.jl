module BeamTrackingDifferentiationInterfaceExt

import BeamTracking: BeamTracking
using BeamTracking: BeamTrackingFlow, AutoBeamTracking, Bunch, track!

import DifferentiationInterface as DI

DI.check_available(::AutoBeamTracking) = true

function DI.prepare_jacobian_nokwarg(
        strict::Val, f::BeamTrackingFlow, backend::AutoBeamTracking, x, contexts::Vararg{DI.Context, C}
    ) where {C}
  _sig = DI.signature(f, backend, x, contexts...; strict)
  return DI.NoJacobianPrep(_sig)
end

function _track_value_and_jac!(flow::BeamTrackingFlow, x::AbstractVector)
  length(x) == 6 || throw(ArgumentError("BeamTrackingFlow expects length(x)==6"))
  v = reshape([float(x[i]) for i in 1:6], 1, 6)
  bunch = Bunch(v=v, species=flow.species, p_over_q_ref=flow.p_over_q_ref, t_ref=flow.t_ref, jacobian=true)
  track!(bunch, flow.kcall_or_chain; flow.track_kw...)
  return bunch, bunch.jac
end

function DI.jacobian(
        flow::BeamTrackingFlow,
        prep::DI.NoJacobianPrep,
        backend::AutoBeamTracking,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
  DI.check_prep(flow, prep, backend, x, contexts...)
  (C > 0) && throw(ArgumentError("BeamTrackingFlow + AutoBeamTracking does not support DifferentiationInterface contexts"))
  _bunch, jac = _track_value_and_jac!(flow, x)
  return Matrix(jac[1, :, :])
end

function DI.value_and_jacobian(
        flow::BeamTrackingFlow,
        prep::DI.NoJacobianPrep,
        backend::AutoBeamTracking,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
  DI.check_prep(flow, prep, backend, x, contexts...)
  (C > 0) && throw(ArgumentError("BeamTrackingFlow + AutoBeamTracking does not support DifferentiationInterface contexts"))
  bunch, jac = _track_value_and_jac!(flow, x)
  return Vector(bunch.v[1, 1:6]), Matrix(jac[1, :, :])
end

function DI.jacobian!(
        flow::BeamTrackingFlow,
        jac,
        prep::DI.NoJacobianPrep,
        backend::AutoBeamTracking,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
  J = DI.jacobian(flow, prep, backend, x, contexts...)
  return copyto!(jac, J)
end

function DI.value_and_jacobian!(
        flow::BeamTrackingFlow,
        jac,
        prep::DI.NoJacobianPrep,
        backend::AutoBeamTracking,
        x,
        contexts::Vararg{DI.Context, C},
    ) where {C}
  y, J = DI.value_and_jacobian(flow, prep, backend, x, contexts...)
  return y, copyto!(jac, J)
end

function BeamTracking.beamtracking_jacobian_flat(flow::BeamTrackingFlow)
  return x -> vec(DI.jacobian(flow, AutoBeamTracking(), x))
end

end # module
