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

end