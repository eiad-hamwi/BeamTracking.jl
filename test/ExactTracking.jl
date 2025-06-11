@testset "ExactTracking" begin
    @testset "Utility functions" begin
        dx_rot = -0.1
        dy_rot = -0.1
        dz_rot = 0.2
  
        W = [cos(dy_rot) 0 sin(dy_rot); 0 1 0; -sin(dy_rot) 0 cos(dy_rot)] *
          [1 0 0; 0 cos(dx_rot) -sin(dx_rot); 0 sin(dx_rot) cos(dx_rot)] *
          [cos(dz_rot) -sin(dz_rot) 0; sin(dz_rot) cos(dz_rot) 0; 0 0 1]
  
        # Test w_matrix function
        @test all(ExactTracking.w_matrix(dx_rot, dy_rot, dz_rot) .== W)
  
        Winv = [cos(dz_rot) sin(dz_rot) 0; -sin(dz_rot) cos(dz_rot) 0; 0 0 1] *
             [1 0 0; 0 cos(dx_rot) sin(dx_rot); 0 -sin(dx_rot) cos(dx_rot)] *
             [cos(dy_rot) 0 -sin(dy_rot); 0 1 0; sin(dy_rot) 0 cos(dy_rot)]
  
        # Test w_inv_matrix function
        @test all(ExactTracking.w_inv_matrix(dx_rot, dy_rot, dz_rot) .== Winv)
    end
  
    @testset "Kernels" begin
        function patch_args(::Type{T}) where {T}
            p0c = T(10e6)
            mc2 = T(ELECTRON.mass)
            tilde_m = mc2/p0c
            gamsqr_0 = 1 + 1/tilde_m^2
            beta_0 = 1/sqrt(1 + tilde_m^2)
            dt = T(1e-9)
            dx = T(2)
            dy = T(3)
            dz = T(4)
            winv = ExactTracking.w_inv_matrix(T(-5),T(6),T(7))
            L = winv[3,1]*dx + winv[3,2]*dy + winv[3,3]*dz
            return L, tilde_m, dt, dx, dy, dz, winv
        end

        function patch_norot_args(::Type{T}) where {T}
            p0c = T(10e6)
            mc2 = T(ELECTRON.mass)
            tilde_m = mc2/p0c
            gamsqr_0 = 1 + 1/tilde_m^2
            beta_0 = 1/sqrt(1 + tilde_m^2)
            dt = T(4e-9)
            dx = T(1)
            dy = T(2)
            dz = T(3)
            L = dz
            return L, tilde_m, dt, dx, dy, dz, nothing
        end

        function drift_args(::Type{T}) where {T}
            L = T(1)
            p0c = T(10e6)
            mc2 = T(ELECTRON.mass)
            tilde_m = mc2/p0c
            gamsqr_0 = 1 + 1/tilde_m^2
            beta_0 = 1/sqrt(1 + tilde_m^2)
            return L, tilde_m, gamsqr_0, beta_0
        end
        
        function solenoid_args(::Type{T}) where {T}
            L = T(1)
            ks = T(2)
            p0c = T(10e6)
            mc2 = T(ELECTRON.mass)
            tilde_m = mc2/p0c
            gamsqr_0 = 1 + 1/tilde_m^2
            beta_0 = 1/sqrt(1 + tilde_m^2)
            return L, ks, tilde_m, gamsqr_0, beta_0
        end

        # Scalar parameters
        test_map(ExactTracking.patch!, "bmad_maps/patch.jl", patch_args(Float64)...; tol=5e-10)
        test_map(ExactTracking.patch!, "bmad_maps/patch_norot.jl", patch_norot_args(Float64)...; tol=1e-9)
        test_map(ExactTracking.exact_drift!, "bmad_maps/drift.jl", drift_args(Float64)...; tol=5e-10)
        test_map(ExactTracking.exact_solenoid!, "bmad_maps/solenoid.jl", solenoid_args(Float64)...; tol=5e-10)

        # GTPSA parameters
        test_map(ExactTracking.patch!, "bmad_maps/patch.jl", patch_args(TPS64{D})...; tol=5e-10, TPS_params=true)
        test_map(ExactTracking.patch!, "bmad_maps/patch_norot.jl", patch_norot_args(TPS64{D})...; tol=1e-9, TPS_params=true)
        test_map(ExactTracking.exact_drift!, "bmad_maps/drift.jl", drift_args(TPS64{D})...; tol=5e-10, TPS_params=true)
        test_map(ExactTracking.exact_solenoid!, "bmad_maps/solenoid.jl", solenoid_args(TPS64{D})...; tol=5e-10, TPS_params=true)

        function comb_func_args(::Type{T}) where {T}
            L = T(1)
            g = T(0.01)
            dg = T(-0.005)
            k1 = T(0.1)
            p0c = T(10e6)
            tilde_m = T(ELECTRON.mass) / p0c
            me1 = T(0.0)
            me2 = T(0.0)
            return L, g, dg, k1, tilde_m, me1, me2
        end

     #   test_map(LinearTracking.combined_func!, "bmad_maps/comb_func.jl", comb_func_args(Float64)...; tol=1e-6)

    end 
    
end