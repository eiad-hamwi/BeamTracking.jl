module BeamTrackingCUDAExt
using CUDA: CuDeviceArray
import BeamTracking: gaussian_random

"""
This function returns two Gaussian random numbers with 
mean 0 and standard deviations sigma1, sigma2 using a 
Box-Muller transform.

This was implemented because CUDA.randn has some horrible 
compiler bug, but CUDA.rand seems to be ok.
"""
function gaussian_random(::CuDeviceArray, sigma1, sigma2)
  s, c = sincospi(2 * rand())
  t = sqrt(-2 * log(rand()))
  z0 = c*t*sigma1
  z1 = s*t*sigma2
  return z0, z1
end

end