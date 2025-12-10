@def_integrator_struct(Implicit)

module ImplicitTracking
using ..LinearAlgebra, ..ForwardDiff
using ..GTPSA, ..BeamTracking, ..StaticArrays, ..KernelAbstractions, ..SIMDMathFunctions
using ..BeamTracking: XI, PXI, YI, PYI, ZI, PZI, Q0, QX, QY, QZ, STATE_ALIVE, STATE_LOST, @makekernel, Coords, vifelse, C_LIGHT

# ==============================================================================
# 0. Math Helpers
# ==============================================================================


# Branch-Free 3x3 Inverse (Same as before)
@inline function solve_3x3_simd(A::SMatrix{3,3,T}, b::SVector{3,T}) where T
    c11 = A[2,2]*A[3,3] - A[2,3]*A[3,2]; c12 = A[2,3]*A[3,1] - A[2,1]*A[3,3]; c13 = A[2,1]*A[3,2] - A[2,2]*A[3,1]
    c21 = A[1,3]*A[3,2] - A[1,2]*A[3,3]; c22 = A[1,1]*A[3,3] - A[1,3]*A[3,1]; c23 = A[1,2]*A[3,1] - A[1,1]*A[3,2]
    c31 = A[1,2]*A[2,3] - A[1,3]*A[2,2]; c32 = A[1,3]*A[2,1] - A[1,1]*A[2,3]; c33 = A[1,1]*A[2,2] - A[1,2]*A[2,1]
    det = A[1,1]*c11 + A[1,2]*c12 + A[1,3]*c13
    inv_det = one(T) / det
    return SVector{3, T}(
        (c11*b[1] + c21*b[2] + c31*b[3]) * inv_det,
        (c12*b[1] + c22*b[2] + c32*b[3]) * inv_det,
        (c13*b[1] + c23*b[2] + c33*b[3]) * inv_det
    )
end


# Wrapper for H logic to pass to eval_val_grad_hess
# We use a functor struct or closure. Closure is fine if inlined.
@inline function get_H_closure(A_func, J_func, tilde_m, β0)
    return w -> begin
        r = SVector(w[1], w[3])
        a = evaluate_field_with_chain_rule(A_func, J_func, r)
        rel_p_z = 1 + w[6]
        rx = w[2] - a[1]
        ry = w[4] - a[2]
        R = rel_p_z^2 - rx^2 - ry^2
        return -sqrt(R) - a[3] + sqrt(rel_p_z^2 + tilde_m^2) / β0
    end
end


"""
norm_sq(::SVector{3,T})::T
calculates dot(r,r) using Fused Multiply-Add
"""
@inline norm_sq(v::SVector{3,T}) where T = fma(v[1], v[1], fma(v[2], v[2], v[3] * v[3]))



# ==============================================================================
# 1. Gradient Interface
# ==============================================================================

"""
Helper to compute gradients (∂H/∂q, ∂H/∂p).
If a user_grad is provided, use it. Otherwise, use AD on H.
"""
@inline function ∇q_H(A, q, p, ::Nothing)
    # Use ForwardDiff if no user gradient is provided.
    # Note: For optimal performance with ForwardDiff, inputs should be StaticArrays 
    # or small vectors. We define a closure for the full gradient.
    TotalGrad = ForwardDiff.gradient(x -> H(x[1], x[2]), (q, p))
    return (TotalGrad[1], TotalGrad[2])
end

"""
Function to construct ∇ₓH from ∇ₓAᵘ
Note that ∂H/∂q₃ corresponds to ∂H/∂z which is derived with chain rule by multiplying ∂Aₛ/∂t and ∂ϕ/∂t by ∂t/∂z = 1 / (βc).
"""
@inline function ∇q_H(A::F, q, p, s, g, β0, tilde_m2, Jac::J) where {F, J}
    # User provided gradient function expected to return (dq, dp)
    T = eltype(q)
    p_x, p_y, p_z = p[1], p[2], p[3]
    x, y, z       = q[1], q[2], q[3]
    rel_p = 1 + p_z
    tilde_E2 = fma(rel_p, rel_p, tilde_m2)

    β_inv =  β0 * sqrt(tilde_E2)
    β = rel_p / β_inv

    t = z / ( β * C_LIGHT )
    A_val =   A(x, y, s, t)
    J_val = Jac(x, y, s, t)

    κ = fma(g, x, 1)      # 1 + g x

    p_kin_x = p_x - A_val[2]
    p_kin_y = p_y - A_val[3]
    p_kin_2 = fma(p_kin_x, p_kin_x, p_kin_y * p_kin_y)

    p_s2 = fma(rel_p, rel_p, - p_kin_2)
    p_s = sqrt(vifelse(p_s2 > 0, p_s2, one(p_s2)))
    scale = κ / p_s

    ∂qx_H = - κ * J_val[4,1] - J_val[1,1] - scale * fma( p_kin_x, J_val[2,1], p_kin_y * J_val[3,1] ) - g * p_s
    ∂qy_H = - κ * J_val[4,2] - J_val[1,2] - scale * fma( p_kin_x, J_val[2,2], p_kin_y * J_val[3,2] )
    ∂qz_H = - κ * J_val[4,3] - J_val[1,3] - scale * fma( p_kin_x, J_val[2,3], p_kin_y * J_val[3,3] )

    return SVector{3,T}(∂qx_H, ∂qy_H, ∂qz_H)
end


@inline function ∇p_H(A::F, q, p, s, g, β0, γ0_2, tilde_m2) where {F}
    # User provided gradient function expected to return (dq, dp)
    T = eltype(p)
    p_x, p_y, p_z = p[1], p[2], p[3]
    x, y, z       = q[1], q[2], q[3]
    rel_p = 1 + p_z
    rel_p2 = rel_p * rel_p
    tilde_E = fma(rel_p, rel_p, tilde_m2)

    β_inv =  β0 * sqrt(tilde_E)
    β = rel_p / β_inv

    t = z / ( β * C_LIGHT )
    A_val = A(x, y, s, t)

    κ = fma(g, x, 1)      # 1 + g x
    κ2 = fma(g, x, 2)     # 2 + g x

    p_kin_x = p_x - A_val[2]
    p_kin_y = p_y - A_val[3]
    p_kin_2 = fma(p_kin_x, p_kin_x, p_kin_y * p_kin_y)

    p_s2 = fma(rel_p, rel_p, - p_kin_2)
    p_s = sqrt(vifelse(p_s2 > 0, p_s2, one(p_s2)))
    scale = κ / p_s

    ∂px_H = scale * p_kin_x
    ∂py_H = scale * p_kin_y
    ∂pz_H  = β - scale * rel_p
            # 
            # =rel_p * (
            #         fma(rel_p2, fma(-g * x, κ2, 1 / γ0_2), - fma((κ * β0) ^ 2, tilde_m2, p_kin_2))
            # ) / (
            #     β_inv * p_s * fma(β_inv, κ, p_s) 
            # )
            #
            # =rel_p * (
            #     (1 / γ0_2 - g * x * (2 + g * x)) * rel_p2 - (p_kin_2 + tilde_m2 * (κ * β0) ^ 2)
            # ) / (
            #     β_inv * p_s * fma(β_inv, κ, p_s) 
            # )

    return SVector{3,T}(∂px_H, ∂py_H, ∂pz_H)
end

# ==============================================================================
# 2. Solver Interface
# ==============================================================================

"""
Generic skeleton for finding roots of F(x) = 0.
Dispatches based on `method`.

Using type parameters (F, T) instead of ::Function allows Julia to specialize
on the exact closure type, eliminating function barriers and enabling better
inlining and optimization.
"""
function find_root(residual_func::F, x_guess::T, method::Val, abstol=1e-15, reltol=1e-14, max_iter = 20) where {F, T}
    error("Solver method not implemented")
end

"""
Method A: Derivative-Free (e.g., Powell's Dog Leg, Anderson Acceleration).
Does not require the Jacobian of the residual function.
"""
function find_root(f::F, x_guess::StaticVector{3, T}, method::Val{:DerivativeFree}, abstol=1e-15, reltol=1e-14, max_iter = 20) where {F, T}
    # Broyden's "Good" Method (rank-1 update on inverse Jacobian) with damped fallback.
    # This is robust and stays compatible with SIMD/GPU element types.
    
    # 1. Initialization
    x = SVector{3, T}(x_guess)
    r = f(x)
    B = one(SMatrix{3, 3, T})               # inverse-Jacobian seed
    
    # Convert scalars to type T (vital for SIMD/GPU compatibility)
    abs_tol = T(abstol)
    rel_tol = T(reltol)
    step_tol = T(1e-32)
    damping = T(0.5)
    valN = Val(3)

    for _ in 1:max_iter
        # 2. Convergence Check
        norm_x = sqrt(norm_sq(x))
        tol = fma(norm_x, rel_tol, abs_tol)
        tol_sq = tol * tol

        res_sq = norm_sq(r)
        active = res_sq > tol_sq
        if !any(active)
            break
        end

        # 3. Proposed Step (Full Broyden/Newton Step)
        delta = B * r
        x_trial = x - delta
        r_trial = f(x_trial)
        res_trial_sq = norm_sq(r_trial)

        # Check who improved
        improved = res_trial_sq < res_sq
        needs_relax = !improved

        # 4. Conditional Damped Step (Fallback)
        x_used = x_trial
        r_used = r_trial
        
        # SIMD Optimization: Only compute fallback if at least one lane needs it
        if any(needs_relax)
            relax_step = damping * delta    # Damping the direction, not the residual
            x_relax = x - relax_step
            r_relax = f(x_relax)
            
            # Blend results: take trial if improved, else take relaxed
            x_used = SVector{3, T}(ntuple(j -> vifelse(improved, x_trial[j], x_relax[j]), valN))
            r_used = SVector{3, T}(ntuple(j -> vifelse(improved, r_trial[j], r_relax[j]), valN))
        end

        # 5. Broyden's "Good" Update for Inverse Jacobian
        # Update B to approximate the Jacobian better for the next step
        s = x_used - x
        y = r_used - r
        
        By = B * y
        stB = transpose(s) * B       # Row vector (s^T * B)
        stBy = dot(stB, y)           # Scalar (s^T * B * y)
        
        # Protect against division by zero
        inv_denom = vifelse(abs(stBy) > step_tol, T(1) / stBy, zero(T))
        
        # Formula: B_new = B + (s - B*y) * s^T * B / (s^T * B * y)
        update = (s - By) * stB * inv_denom
        B = B + update

        # 6. Update State
        x = x_used
        r = r_used
    end

    return x
end

"""
Method B: Newton-like methods (e.g., Newton-Raphson, Steffensen).
Uses AD to calculate the Jacobian of the *residual function* itself.
"""
function find_root(f::F, x_guess::T, method::Val{:Newton}, abstol::Float64=1e-15, reltol::Float64=1e-14, max_iter = 20) where {F, T}
    # We construct the Jacobian of the residual equation automatically.
    # Note: Even if the inner gradient of H was explicit, we can AD through 
    # the residual function to get the Hessian-vector product required here.
    df(x) = ForwardDiff.jacobian(f, x)
    
    x_curr = x_guess
    # while !converged
    #    J = df(x_curr)
    #    x_curr = x_curr - J \ f(x_curr)
    # end
    return x_curr
end


# ==============================================================================
# 3. Main Integrator Step
# ==============================================================================

"""
Performs a single time-reversible step consisting of two canonical transformations.

Inputs:
- q0, p0: State vectors
- H: Hamiltonian function H(q, p)
- ds: Step size
- solver_method: Val{:DerivativeFree} or Val{:Newton}
- user_grad: Optional function (q,p) -> (∇qH, ∇pH). Pass `nothing` to use AD.
"""
@makekernel fastgtpsa=true function symplectic_step!(i, coords::Coords, 
                                                    A, s, ds, g, β0, γ0_2, tilde_m2, 
                                                    solver_method::Val, user_grad=nothing, 
                                                    abstol=1e-15, reltol=1e-14, max_iter=50,
                                                    )
    v = coords.v
    T = eltype(v[i,:])
    q0 = SVector{3,T}(v[i, XI], v[i, YI], v[i, ZI])
    p0 = SVector{3,T}(v[i, PXI], v[i, PYI], v[i, PZI])
    ds2 = ds / 2

    # ======================================================
    # --- Transformation 1 ---
    # Implicit relation: q1 = q0 + ∇p H(q1, p0) * ds
    # We solve for q1.
    
    # function residual_step_1(q_trial)
    #     # Calculate gradients at (q_trial, p0)
    #     # Note: We only need ∇p here.
    #     dHdp = grad_p_H(H, q_trial, p0, user_grad)
    #     return q_trial - q0 - dHdp * ds
    # end

    # Explicit update for p1 using the resolved q1
    # p1 = p0 - ∇q H(q1, p0) * ds


    q1 = find_root(
        (q1) -> q1 - q0 - ds2 * ∇p_H(A, q1, p0,
                                     s, g, β0, 
                                     γ0_2, 
                                     tilde_m2),
        q0, solver_method, abstol=abstol, reltol=reltol, max_iter=max_iter
    )

    p1 = p0 - ds2 * ∇q_H(A, q1, p0, s, g, β0, tilde_m2, user_grad)


    # ======================================================
    # --- Transformation 2 ---
    # Implicit relation: p2 = p1 - ∇q H(q1, p2) * ds
    # We solve for p2. Note that q1 is now fixed from the previous step.
    
    # function residual_step_2(p_trial)
    #     # Calculate gradients at (q1, p_trial)
    #     dHdq = grad_q_H(H, q1, p_trial, user_grad)
    #     return p_trial - p1 + dHdq * ds
    # end

    # Explicit update for q2
    # q2 = q1 + ∇p H(q1, p2) * ds


    p2 = find_root(
        (p2) -> p2 - p1 + ds2 * ∇q_H(A, q1, p2,
                                     s, g, β0, 
                                     tilde_m2, user_grad), 
        p1, solver_method, abstol=abstol, reltol=reltol, max_iter=max_iter
    )

    q2 = q1 + ds2 * ∇p_H(A, q1, p2, s, g, β0, γ0_2, tilde_m2)

    v[i,  XI] = q2[1]
    v[i, PXI] = p2[1]
    v[i,  YI] = q2[2]
    v[i, PYI] = p2[2]
    v[i,  ZI] = q2[3]
    v[i, PZI] = p2[3]
end

end