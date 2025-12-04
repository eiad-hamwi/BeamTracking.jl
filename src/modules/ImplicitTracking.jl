@def_integrator_struct(Implicit)

module ImplicitTracking
using ..LinearAlgebra, ..ForwardDiff
using ..GTPSA, ..BeamTracking, ..StaticArrays, ..KernelAbstractions, ..SIMDMathFunctions
using ..BeamTracking: XI, PXI, YI, PYI, ZI, PZI, Q0, QX, QY, QZ, STATE_ALIVE, STATE_LOST, @makekernel, Coords, vifelse, C_LIGHT

# ==============================================================================
# 1. Math Helpers: Register-Based Dual Extraction
# ==============================================================================

"""
    eval_val_grad_hess(f, x::SVector{N, T})

Evaluates `f(x)` using nested ForwardDiff Duals to compute Value, Gradient, and Hessian
in a single pass. Returns (val, grad, hess) as SVector/SMatrix types.
"""
@inline function eval_val_grad_hess(f::F, x::SVector{N, T}) where {F, N, T}
    # 1. Define Tags for safety (prevents perturbation confusion)
    Tag1 = typeof(ForwardDiff.Tag(f, T))
    Tag2 = typeof(ForwardDiff.Tag(f, ForwardDiff.Dual{Tag1, T, N}))

    # 2. Construct Inner Duals (Layer 1: Gradient)
    # x_i -> Dual(x_i, e_i) where e_i is the standard basis
    # We use a generator to build the SVector directly.
    x_dual_1 = SVector{N}(ntuple(i -> ForwardDiff.Dual{Tag1, T, N}(x[i], ForwardDiff.Partials{N, T}(ntuple(j -> i==j ? one(T) : zero(T), Val(N)))), Val(N)))

    # 3. Construct Outer Duals (Layer 2: Hessian)
    # Wrap Layer 1 in another layer of Duals with identity partials
    x_dual_2 = SVector{N}(ntuple(i -> ForwardDiff.Dual{Tag2, eltype(x_dual_1), N}(x_dual_1[i], ForwardDiff.Partials{N, eltype(x_dual_1)}(ntuple(j -> i==j ? one(eltype(x_dual_1)) : zero(eltype(x_dual_1)), Val(N)))), Val(N)))

    # 4. Evaluate Function ONCE
    result = f(x_dual_2)

    # 5. Unpack Results purely from registers
    
    # Layer 2 Value is a Dual (containing Layer 1 info)
    res_val_L2 = ForwardDiff.value(result) 
    
    # Layer 1 Value is the actual scalar function value
    val = ForwardDiff.value(res_val_L2)
    
    # Layer 1 Partials is the Gradient
    grad = SVector{N, T}(res_val_L2.partials.values)

    # Layer 2 Partials contains the Hessian rows (as Duals)
    # We extract the partials of those Duals to get the T values.
    hess_rows = ForwardDiff.partials(result).values # Tuple of N Duals
    
    # Construct SMatrix from the partials of the Duals in hess_rows
    hess = SMatrix{N, N, T}(ntuple(j -> ntuple(i -> hess_rows[j].partials.values[i], Val(N)), Val(N))...)
    
    # Note on Hessian Layout:
    # hess_rows[j] is the partial derivative w.r.t input j (Outer Layer)
    # hess_rows[j].partials[i] is the partial of that w.r.t input i (Inner Layer)
    # So this builds the matrix naturally. Transpose check: Hessian is symmetric, so safe.

    return val, grad, hess
end

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

# Recursive Chain Rule (Same as before)
@inline function evaluate_field_with_chain_rule(A_func, J_func, r::SVector{N, T}) where {N, T<:Number}
    return A_func(r)
end

@inline function evaluate_field_with_chain_rule(A_func, J_func, r::SVector{N, D}) where {N, D<:ForwardDiff.Dual}
    r_val = ForwardDiff.value.(r)
    A_val = evaluate_field_with_chain_rule(A_func, J_func, r_val)
    if isnothing(J_func)
        return A_func(r)
    else
        J_mat = J_func(r_val)
        # Chain rule: NewPartial = J * OldPartial
        r_part = ForwardDiff.partials.(r)
        # Unrolled Matrix-Vector mul for Partials
        p_1 = J_mat[1,1] * r_part[1] + J_mat[1,2] * r_part[2]
        p_2 = J_mat[2,1] * r_part[1] + J_mat[2,2] * r_part[2]
        p_3 = J_mat[3,1] * r_part[1] + J_mat[3,2] * r_part[2]
        return SVector{3, D}(D(A_val[1], p_1), D(A_val[2], p_2), D(A_val[3], p_3))
    end
end

# ==============================================================================
# 2. Main Solvers (Implicit)
# ==============================================================================

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

@makekernel fastgtpsa=true function implicit_full_step!(i, coords::Coords, A_func, tilde_m, β0, ds, J_func=nothing)
    implicit_solve_P!(i, coords.v, A_func, J_func, tilde_m, β0, ds/2)
    implicit_solve_Q!(i, coords.v, A_func, J_func, tilde_m, β0, ds/2)
end

@inline function implicit_solve_P!(i, v, A_func, J_func, tilde_m, β0, ds)
    T = eltype(v)
    
    # 1. Setup State
    q_old = SVector{3, T}(v[i, XI], v[i, YI], v[i, ZI])
    p_old = SVector{3, T}(v[i, PXI], v[i, PYI], v[i, PZI])
    p_curr = p_old
    
    # 2. Setup Hamiltonian Closure
    H_func = get_H_closure(A_func, J_func, tilde_m, β0)

    # 3. Newton Loop
    tol_sq = T(1e-28) # 1e-14 squared
    max_iter = 7
    
    for _ in 1:max_iter
        z = SVector{6, T}(q_old[1], p_curr[1], q_old[2], p_curr[2], q_old[3], p_curr[3])
        
        # --- The Optimized Call ---
        _, g, H = eval_val_grad_hess(H_func, z)
        # --------------------------
        
        # Res = p_curr - p_old + ds * dH/dq  (dH/dq indices: 1, 3, 5)
        dH_dq = SVector(g[1], g[3], g[5])
        res = p_curr - p_old + ds * dH_dq
        
        res_sq = res[1]^2 + res[2]^2 + res[3]^2
        active = res_sq > tol_sq
        if !any(active) break end

        # J = I + ds * d(dH/dq)/dp  (Row: 1,3,5; Col: 2,4,6)
        H_qp = SMatrix{3,3,T}(H[1,2], H[3,2], H[5,2], H[1,4], H[3,4], H[5,4], H[1,6], H[3,6], H[5,6])
        J = I + ds * H_qp
        
        delta = solve_3x3_simd(J, res)
        p_new = p_curr - delta
        
        p_curr = SVector{3,T}(
            vifelse(active, p_new[1], p_curr[1]),
            vifelse(active, p_new[2], p_curr[2]),
            vifelse(active, p_new[3], p_curr[3])
        )
    end
    
    # 4. Final Update (Q)
    z_final = SVector{6, T}(q_old[1], p_curr[1], q_old[2], p_curr[2], q_old[3], p_curr[3])
    _, g_final, _ = eval_val_grad_hess(H_func, z_final)
    
    q_new = q_old + ds * SVector(g_final[2], g_final[4], g_final[6])
    
    v[i, XI], v[i, YI], v[i, ZI] = q_new
    v[i, PXI], v[i, PYI], v[i, PZI] = p_curr
end

@inline function implicit_solve_Q!(i, v, A_func, J_func, tilde_m, β0, ds)
    T = eltype(v)
    
    q_old = SVector{3, T}(v[i, XI], v[i, YI], v[i, ZI])
    p_old = SVector{3, T}(v[i, PXI], v[i, PYI], v[i, PZI])
    q_curr = q_old
    
    H_func = get_H_closure(A_func, J_func, tilde_m, β0)
    tol_sq = T(1e-28)
    max_iter = 7
    
    for _ in 1:max_iter
        z = SVector{6, T}(q_curr[1], p_old[1], q_curr[2], p_old[2], q_curr[3], p_old[3])
        
        _, g, H = eval_val_grad_hess(H_func, z)
        
        # Res = q_curr - q_old - ds * dH/dp (dH/dp indices: 2, 4, 6)
        dH_dp = SVector(g[2], g[4], g[6])
        res = q_curr - q_old - ds * dH_dp
        
        res_sq = res[1]^2 + res[2]^2 + res[3]^2
        active = res_sq > tol_sq
        if !any(active) break end

        # J = I - ds * d(dH/dp)/dq (Row: 2,4,6; Col: 1,3,5)
        H_pq = SMatrix{3,3,T}(H[2,1], H[4,1], H[6,1], H[2,3], H[4,3], H[6,3], H[2,5], H[4,5], H[6,5])
        J = I - ds * H_pq
        
        delta = solve_3x3_simd(J, res)
        q_new = q_curr - delta
        
        q_curr = SVector{3,T}(
            vifelse(active, q_new[1], q_curr[1]),
            vifelse(active, q_new[2], q_curr[2]),
            vifelse(active, q_new[3], q_curr[3])
        )
    end
    
    z_final = SVector{6, T}(q_curr[1], p_old[1], q_curr[2], p_old[2], q_curr[3], p_old[3])
    _, g_final, _ = eval_val_grad_hess(H_func, z_final)
    
    p_new = p_old - ds * SVector(g_final[1], g_final[3], g_final[5])
    
    v[i, XI], v[i, YI], v[i, ZI] = q_curr
    v[i, PXI], v[i, PYI], v[i, PZI] = p_new
end
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================











# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================
# ==============================================================================













# ==============================================================================
# 1. Gradient Interface (Bifurcation 1)
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

@inline function ∇q_H(A::F, q, p, s, t, ds, tilt_ref, g, β0, tilde_m2) where {F}
    # User provided gradient function expected to return (dq, dp)
    p_x, p_y, p_z = p[1], p[2], p[3]
    q_x, q_y, q_z = q[1], q[2], q[3]
    rel_p = 1 + p_z

    β_inv =  β0 * sqrt(muladd(rel_p, rel_p, tilde_m2))
    β = rel_p / β_inv
    t = q_z / ( β * C_LIGHT )
    A_val = A((q_x, q_y, s, t)) 
    κ = 1 + q_x * g * cos(tilt_ref)

    p_kin_x = p_x - A_val[1]
    p_kin_y = p_y - A_val[2]
    p_kin_2 = muladd(p_kin_x, p_kin_x, p_kin_y * p_kin_y)

    p_s2 = muladd(rel_p, rel_p, - p_kin_2)
    p_s = sqrt(vifelse(p_s2 > 0, p_s2, one(p_s2)))
    scale = κ / p_s

    ∂p_x_H = scale * p_kin_x
    ∂p_y_H = scale * p_kin_y
    ∂pz_H  = scale * rel_p - β
#=
        (1 + g * x) * (1 + δ) / sqrt(rel_p^2 - p_kin_2) - 1 / (β0 sqrt((1 + δ)^2 + tilde_m^2))
        ((κ^2 * rel_p^2 * β0^2 - 1) * rel_p^2 + κ^2 * rel_p^2 * β0^2 * tilde_m^2 + p_kin_2 ) / ( p_s * β_inv * (κ * β_inv + p_s ) )

           = rel_p * (p_kin_2 - p_z * (2 + p_z) / gamsqr_0)
                / ( β_inv * p_s * (β_inv + p_s) )
=#
    return SVector(∂p_x_H, ∂p_y_H, ∂pz_H)
end


@inline function ∇p_H(A::F, q, p, g, β0, tilde_m) where {F}
    # User provided gradient function expected to return (dq, dp)
    κ = 1 + g * q[1]
    p_kin = p[1:2] .- A([q[1], q[2]])

    rel_p = 1 + p[3]
    p_s2 = rel_p * rel_p - p_kin[1] * p_kin[1] - p_kin[2] * p_kin[2]
    p_s = sqrt(vifelse(p_s2 > 0, p_s2, one(p_s2)))

    ∇⟂H = κ * p_kin / p_s

    ∂pz_H = κ / p_s + 1 / ( β0 * sqrt(rel_p * rel_p + tilde_m2) )
    ∂pz_H = rel_p * ∂pz_H

    return SVector(∇⟂H..., ∂pz_H)
end

# ==============================================================================
# 2. Solver Interface (Bifurcation 2)
# ==============================================================================

"""
Generic skeleton for finding roots of F(x) = 0.
Dispatches based on `method`.

Using type parameters (F, T) instead of ::Function allows Julia to specialize
on the exact closure type, eliminating function barriers and enabling better
inlining and optimization.
"""
function find_root(residual_func::F, x_guess::T, method::Val{M}) where {F, T, M}
    error("Solver method not implemented")
end

"""
Method A: Derivative-Free (e.g., Powell's Dog Leg, Anderson Acceleration).
Does not require the Jacobian of the residual function.
"""
function find_root(f::F, x_guess::StaticVector{3, T}, method::Val{:DerivativeFree}; abstol=1e-12, reltol=1e-12) where {F, T}
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
    max_iter = 10

    for _ in 1:max_iter
        # 2. Convergence Check
        norm_x = sqrt(dot(x, x))
        tol = abs_tol + rel_tol * norm_x
        tol_sq = tol * tol

        res_sq = dot(r, r)
        active = res_sq > tol_sq
        if !any(active)
            break
        end

        # 3. Proposed Step (Full Broyden/Newton Step)
        delta = B * r
        x_trial = x - delta
        r_trial = f(x_trial)
        res_trial_sq = dot(r_trial, r_trial)
        
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
function find_root(f::F, x_guess::T, method::Val{:Newton}; abstol::Float64=1e-12, reltol::Float64=1e-12) where {F, T}
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
@makekernel fastgtpsa=true function symplectic_step!(i, coords::Coords, A, ds, solver_method::Val{M}, user_grad = nothing) where M
    v = coords.v
    q0 = SVector(v[i, XI], v[i, YI], v[i, ZI])
    p0 = SVector(v[i, PXI], v[i, PYI], v[i, PZI])
    
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


    q1 = find_root((q1) -> q1 - q0 - ∇p_H(A, q1, p0, user_grad) * ds, q0, solver_method)

    dHdq_1 = ∇q_H(A, q1, p0, user_grad)
    p1 = p0 - dHdq_1 * ds


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

    
    p2 = find_root((p2) -> p2 - p1 + ∇q_H(A, q1, p2, user_grad) * ds, p1, solver_method)

    dHdp_2 = ∇p_H(A, q1, p2)
    q2 = q1 + dHdp_2 * ds

    return q2, p2
end

end