using Test,
      BeamTracking,
      GTPSA,
      LinearAlgebra,
      StaticArrays,
      SIMD

using BeamTracking: launch!, Coords, KernelCall, XI, YI, ZI, PXI, PYI, PZI, C_LIGHT


const D1 = Descriptor(6, 1)   # 6 variables 1st order
const D2 = Descriptor(6, 2)   # 6 variables 2nd order
const D10 = Descriptor(6, 10) # 6 variables 10th order

zero_d = zero(eltype(@vars(D1)[1]))
one_d = one(eltype(@vars(D1)[1]))




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


function H(q, p;
        params...)
    A = params[:A] # (x,y,s,t)->[0,0,0,0]
    g = params[:g] # 0
    beta_0 = params[:beta_0] # 1
    tilde_m = params[:tilde_m] # 0

    a = A(q[1], q[2], zero(q[1]), zero(q[1]))
    p_kin_x = p[1] - a[2]
    p_kin_y = p[2] - a[3]
    kinetic_term = -(1+g*q[1]) * sqrt((1+p[3])^2 - p_kin_x^2 - p_kin_y^2)
    magnetic_term = -(1+g*q[1]) * a[4]
    electric_term = -a[1]
    reference_term = sqrt((1+p[3])^2 + tilde_m^2) / beta_0
    return kinetic_term + magnetic_term + electric_term + reference_term
end



function F1(q, p0; H, ds2, params...)
    q0 = params[:q0]
    ∇H = H(q, p0+@vars(D1)[[2,4,6]]; params...)
    ∇_p_H = [
        ∇H[[0,1,0,0,0,0]],
        ∇H[[0,0,0,1,0,0]],
        ∇H[[0,0,0,0,0,1]]
    ]
    return q - q0 - ds2 * ∇_p_H
end

function F2(q0, p; H, ds2,params...)
    p0 = params[:p0]
    ∇H = H(q0+@vars(D1)[[1,3,5]], p; params...)
    ∇_q_H = [
        ∇H[[1,0,0,0,0,0]],
        ∇H[[0,0,1,0,0,0]],
        ∇H[[0,0,0,0,1,0]]
    ]
    return p - p0 + ds2 * ∇_q_H
end

function JF1(q, p0; H, ds2, params...)
    ∇H = H(q+@vars(D2)[[1,3,5]], p0+@vars(D2)[[2,4,6]]; params...)
    ∇q_∇p_H = [
        ∇H[[1,1,0,0,0,0]] ∇H[[0,1,1,0,0,0]] ∇H[[0,1,0,0,1,0]];
        ∇H[[1,0,0,1,0,0]] ∇H[[0,0,1,1,0,0]] ∇H[[0,0,0,1,1,0]];
        ∇H[[1,0,0,0,0,1]] ∇H[[0,0,1,0,0,1]] ∇H[[0,0,0,0,1,1]]
    ]
    return I - ds2 * ∇q_∇p_H
end
        
function JF2(q0, p; H, ds2, params...)
    ∇H = H(q0+@vars(D2)[[1,3,5]], p+@vars(D2)[[2,4,6]]; params...)
    ∇p_∇q_H = [
        ∇H[[1,1,0,0,0,0]] ∇H[[1,0,0,1,0,0]] ∇H[[1,0,0,0,0,1]];
        ∇H[[0,1,1,0,0,0]] ∇H[[0,0,1,1,0,0]] ∇H[[0,0,1,0,0,1]];
        ∇H[[0,1,0,0,1,0]] ∇H[[0,0,0,1,1,0]] ∇H[[0,0,0,0,1,1]]
    ]
    return I + ds2 * ∇p_∇q_H
end

function find_root(F, JF, x, abstol=1e-15, reltol=1e-14, max_iter=20)
    function converged(x)
        return norm(F(x)) < abstol + reltol * norm(x)
    end
    for _ in 1:max_iter if converged(x) break end
        x = x - JF(x) \ F(x)
    end
    return x
end



@testset "ImplicitIntegration" begin
    @testset "Plain Implicit Implementation" begin
        q0 = [0.0031, -0.0012, 2.312e-8]
        p0 = [0.001, -0.002, 0.001355]
        
        ds2 = 0.013131
        k = 0.23512
        
        p = Species("proton")
        p0c = 131.2e6
        tilde_m, gamsqr_0, beta_0 = ExactTracking.drift_params(p, p0c / C_LIGHT)
        params = (q0=q0, p0=p0, A=A_sol(k)[1], g=0.0, beta_0=beta_0, tilde_m=tilde_m)
        
        b = Bunch([q0[1] p0[1] q0[2] p0[2] q0[3] p0[3]], species=p, R_ref=p0c / C_LIGHT)

        # =============================== SOLENOID TEST ===============================
        b1 = deepcopy(b);
        kc = KernelCall(ExactTracking.exact_solenoid!, (k, beta_0, gamsqr_0, tilde_m, ds2))
        launch!(b1.coords, kc);

        kc = KernelCall(ImplicitTracking.symplectic_step!, (
            A=A_sol(k)[1], s=0.0, ds=ds2, g=0.0, beta_0=beta_0, gamsqr_0=gamsqr_0, tilde_m2=tilde_m^2, 
            solver_method=Val(:DerivativeFree), user_grad=A_sol(k)[2], 
            abstol=1e-15, reltol=1e-14, max_iter=15
        ));
        b2 = deepcopy(b);
        launch!(b2.coords, kc);

        @test norm(b1.coords.v - b2.coords.v) ≈ 0 atol=1e-14

        println("Plain Implicit Solenoid Kick")
        println("Δq = ", find_root((q) -> F1(q, p0; H, ds2, params...), (q) -> JF1(q, p0; H, ds2, params...), deepcopy(q0)) - q0)
        println("Δp = ", find_root((p) -> F2(q0, p; H, ds2, params...), (p) -> JF2(q0, p; H, ds2, params...), deepcopy(p0)) - p0)


        # =============================== DIPOLE TEST ===============================
        g = 0.0123
        A_dipole(g) = (x, y, s, t) -> [0, 0, 0, -g * x]
        JA_dipole(g) = (x, y, s, t) -> [0 0 0; 
                                 0 0 0; 
                                 0 0 0; 
                                -g 0 0]
        b3 = deepcopy(b);
        kc = KernelCall(ImplicitTracking.symplectic_step!, (
            A=A_dipole(g), s=0.0, ds=ds2, g=g, beta_0=beta_0, gamsqr_0=gamsqr_0, tilde_m2=tilde_m^2, 
            solver_method=Val(:DerivativeFree), user_grad=JA_dipole(g), 
            abstol=1e-15, reltol=1e-14, max_iter=15
        ));
        launch!(b3.coords, kc);
        println("--------------------------------")
        println("Implicit Dipole Kick")
        println("Δq = ", b3.coords.v[1,[XI,YI,ZI]] - q0)
        println("Δp = ", b3.coords.v[1,[PXI,PYI,PZI]] - p0)


        params = (q0=q0, p0=p0, A=A_dipole(g), g=g, beta_0=beta_0, tilde_m=tilde_m)
        println("--------------------------------")
        println("Plain Implicit Dipole Kick")
        q1 = find_root((q) -> F1(q, p0; H, ds2, params...), (q) -> JF1(q, p0; H, ds2, params...), deepcopy(q0))
        println("Δq = ", q1 - q0)
        p = find_root((p) -> F2(q1, p; H, ds2, params...), (p) -> JF2(q1, p; H, ds2, params...), deepcopy(p0))
        println("Δp = ", p - p0)


        b4 = deepcopy(b);
        kc = KernelCall(ExactTracking.exact_bend!, (g*ds2, g, g, tilde_m, beta_0, ds2))
        launch!(b4.coords, kc);
        println("--------------------------------")
        println("Exact Dipole Kick")
        println("Δq = ", b4.coords.v[1,[XI,YI,ZI]] - q0)
        println("Δp = ", b4.coords.v[1,[PXI,PYI,PZI]] - p0)

        @test norm(b3.coords.v - b4.coords.v) ≈ 0 atol=1e-12
    end
end
