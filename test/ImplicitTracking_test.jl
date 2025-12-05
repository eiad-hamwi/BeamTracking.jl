using Test,
      BeamTracking,
      GTPSA,
      StaticArrays,
      SIMD

using BeamTracking: launch!, Coords, KernelCall, XI, YI, ZI, PXI, PYI, PZI, C_LIGHT


const D1 = Descriptor(6, 1)   # 6 variables 1st order
const D10 = Descriptor(6, 10) # 6 variables 10th order

zero_d = zero(eltype(@vars(D1)[1]))
one_d = one(eltype(@vars(D1)[1]))


function H(q, p; 
        A=(x,y,s,t)->[0,0,0,0],
        g=0.,
        beta_0=1.,
        tilde_m=0.
        )
    a = A(q[1], q[2], zero(q[1]), zero(q[1]))
    p_kin_x = p[1] - a[2]
    p_kin_y = p[2] - a[3]
    kinetic_term = -(1+g*q[1]) * sqrt((1+p[3])^2 - p_kin_x^2 - p_kin_y^2)
    magnetic_term = -a[4]
    electric_term = -a[1]
    reference_term = sqrt((1+p[3])^2 + tilde_m^2) / beta_0
    return kinetic_term + magnetic_term + electric_term + reference_term
end

function ∇q_H(q, p; params...)
    dx,_,dy,_,dz,_ = @vars(D1)
    grad = H(q + [dx,dy,dz], p; params...)
    return [
        grad[[1,0,0,0,0,0]],
        grad[[0,0,1,0,0,0]],
        grad[[0,0,0,0,1,0]]
    ]
end

function ∇p_H(q, p; params...)
    _,dpx,_,dpy,_,dpz = @vars(D1)
    grad = H(q, p + [dpx,dpy,dpz]; params...)
    return [
        grad[[0,1,0,0,0,0]],
        grad[[0,0,0,1,0,0]],
        grad[[0,0,0,0,0,1]]
    ]
end

function A_quad(k)
    function out(x::T,y::T,s::T,t::T) where {T<:Number} 
        return [0,0,0,-k*(x*x-y*y)/2]
    end
    return out
end

function A_sol(k)
    function func(x::T,y::T,s::T,t::T) where {T<:Number} 
        return [0,-k*y/2,k*x/2,0]
    end
    function jac(x::T,y::T,s::T,t::T) where {T<:Number} 
        return [
        0 0 0 0; 
        0 -k/2 0 0; 
        k/2 0 0 0; 
        0 0 0 0]
    end
    return func, jac
end

@testset "ImplicitIntegration" begin
    @testset "Kernels" begin
        g=0.;
        p = Species("proton")
        R_ref = 121.57
        tilde_m, gamsqr_0, beta_0 = ExactTracking.drift_params(p, R_ref)

        b = Bunch([0.001*one_d 0 0 0 0 0], species=p, R_ref=R_ref)
        v = b.coords.v;
        q = v[1,[XI,YI,ZI]];
        p = v[1,[PXI,PYI,PZI]];

        params = (A=A_quad(0.02), g=g, beta_0=beta_0, tilde_m=tilde_m)
        println(" FOCUSING QUADRUPOLE MAGNETIC FIELD ")
        println("b.coords.v = ", v);
        println("Δp ∝ -∇q_H(q, p) = ", -∇q_H(q, p; params...));
        println("Δq ∝ ∇p_H(q, p) = ", ∇p_H(q, p; params...));

        params = (A=A_sol(0.1), g=g, beta_0=beta_0, tilde_m=tilde_m)
        println(" SOLENOID MAGNETIC FIELD ")
        println("b.coords.v = ", v);
        println("Δp ∝ -∇q_H(q, p) = ", -∇q_H(q, p; params...));
        println("Δq ∝ ∇p_H(q, p) = ", ∇p_H(q, p; params...));

        bt = deepcopy(b);
        vt = bt.coords.v;
        kc = KernelCall(ExactTracking.exact_solenoid!, (0.1, beta_0, gamsqr_0, tilde_m, @vars(D1)[1]));
        launch!(bt.coords, kc);
        println(" EXACT SOLENOID KICKS ");
        println("Δp ∝ ", [vt[1,PXI][1],vt[1,PYI][1],vt[1,PZI][1]]);
        println("Δq ∝ ", [vt[1,XI][1],vt[1,YI][1],vt[1,ZI][1]]);

        bt = deepcopy(b);
        kc = KernelCall(ImplicitTracking.symplectic_step!, (
            A_sol(0.1)[1], 0.0, 0.001, g, beta_0, gamsqr_0, tilde_m^2, 
            solver_method::Val, user_grad=A_sol(0.1)[2], 
            abstol=1e-15, reltol=1e-14, max_iter=25
        ));

        bt = deepcopy(b);
        launch!(bt.coords, kc);
        println(" IMPLICIT SYMPLECTIC STEP ");
        println("Δp = ", [bt.coords.v[1,PXI][1],bt.coords.v[1,PYI][1],bt.coords.v[1,PZI][1]]);
        println("Δq = ", [bt.coords.v[1,XI][1],bt.coords.v[1,YI][1],bt.coords.v[1,ZI][1]]);

    end
end