@makekernel fastgtpsa=true function update_P0!(i, coords::Coords, R_ref_initial, R_ref_final, ramp_without_rf)
  v = coords.v 
  v[i,PXI] = v[i,PXI] * R_ref_initial / R_ref_final
  v[i,PYI] = v[i,PYI] * R_ref_initial / R_ref_final
  if !ramp_without_rf
    v[i,PZI] = (R_ref_initial * (1 + v[i,PZI]) - R_ref_final) / R_ref_final
  end
end
