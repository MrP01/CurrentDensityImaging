module Simulator
import KomaMRI
import Lazy
import GLMakie
import LinearAlgebra
import FastTransforms: fft, ifft
import CurrentDensityImaging.GridPhantom as GP

@kwdef struct CurrentDensityPhantom
  pog::GP.GridPhantom  # phantom on grid
  # j::Vector{Vector{T}} = zeros(size(p.x), 3)
  jx::Array{Float64,3} = ones(size(pog.x))
  jy::Array{Float64,3} = zeros(size(pog.x))
  jz::Array{Float64,3} = zeros(size(pog.x))
end

Lazy.@forward CurrentDensityPhantom.p KomaMRI.plot_phantom_map

function plot_current_density(cdp::CurrentDensityPhantom)
  GLMakie.arrows(cdp.p.x, cdp.p.y, cdp.p.z, cdp.jx, cdp.jy, cdp.jz, axis=(type=Axis3,))
end

function calculate_magnetic_field(cdp::CurrentDensityPhantom)
  mu_0 = 1.0

  g1(k) = -1.0im * k[1] ./ LinearAlgebra.norm(k) .^ 2
  g2(k) = -1.0im * k[2] ./ LinearAlgebra.norm(k) .^ 2
  g3(k) = -1.0im * k[3] ./ LinearAlgebra.norm(k) .^ 2

  s = max(max(cdp.p.x...) - min(cdp.p.x...), max(cdp.p.y...) - min(cdp.p.y...), max(cdp.p.z...) - min(cdp.p.z...))
  N = 100

  k1 = fftfreq(N)
  k = hcat(k1, k1, k1)

  jx_interp = Interpolations.interpolate()
  jx_on_grid = itp()

  B1::Vector{Float64} = mu_0 * ifft(fft(cdp.jy) * g3.(k) - fft(cdp.jz) * g2.(k))
  B2::Vector{Float64} = -mu_0 * ifft(fft(cdp.jx) * g3.(k) - fft(cdp.jz) * g1.(k))
  B3::Vector{Float64} = mu_0 * ifft(fft(cdp.jx) * g2.(k) - fft(cdp.jy) * g1.(k))

  return B1, B2, B3
end
end
