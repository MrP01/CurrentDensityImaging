module CurrentDensityImaging
include("gridphantom.jl")

import KomaMRI
import Lazy
import LinearAlgebra
import FastTransforms: fft, ifft
import .GridPhantom as GP
import GLMakie

@kwdef struct CurrentDensityPhantom
  pog::GP.PhantomOnAGrid  # phantom on grid
  jx::Array{Float64,3} = ones(size(pog.ρ))
  jy::Array{Float64,3} = zeros(size(pog.ρ))
  jz::Array{Float64,3} = zeros(size(pog.ρ))
end

Lazy.@forward CurrentDensityPhantom.p KomaMRI.plot_phantom_map

function plot_current_density(cdp::CurrentDensityPhantom)
  flat = GP.to_flat_phantom(cdp.pog)
  mask = cdp.pog.ρ .!= 0
  GLMakie.arrows(flat.x, flat.y, flat.z, cdp.jx[mask], cdp.jy[mask], cdp.jz[mask], axis=(type=Axis3,))
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

greet() = print("Hello World!")
end
