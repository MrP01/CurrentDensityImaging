module CDI
import KomaMRI
import Lazy
import LinearAlgebra
import FastTransforms: fft, ifft, fftfreq
import GLMakie
import Optim
include("./grid.jl")

const VectorField = Tuple{Array{Float64,3},Array{Float64,3},Array{Float64,3}}

@kwdef struct CurrentDensityPhantom
  pog::grid.PhantomOnAGrid  # phantom on grid
  jx::Array{Float64,3} = ones(size(pog.ρ))
  jy::Array{Float64,3} = zeros(size(pog.ρ))
  jz::Array{Float64,3} = zeros(size(pog.ρ))
end

rtoi(x) = Int32(round(x))

function convertToDemoCDP(pog::grid.PhantomOnAGrid)::CurrentDensityPhantom
  shape = size(pog.ρ)
  jz = zeros(shape)
  jz[rtoi(shape[1] / 3):rtoi(shape[1] * 2 / 3), rtoi(shape[2] / 3):rtoi(shape[2] * 2 / 3), :] .= 1 / 4000
  # jz[4, 4, :] .= 1 / 2000
  return CurrentDensityPhantom(pog, zeros(size(pog.ρ)), zeros(size(pog.ρ)), jz)
end
function generateDemoCDP(shape=(8, 8, 4))::CurrentDensityPhantom
  sero = zeros(shape)
  pog = grid.PhantomOnAGrid(name="demo phantom!", ρ=ones(shape), T1=sero, T2=sero, T2s=sero, Δw=sero, Δx=vec([0.001, 0.001, 0.001]), offset=vec([0, 0, 0]))
  return convertToDemoCDP(pog)
end
function loadBrainCDP()::CurrentDensityPhantom
  brain_h5 = joinpath(dirname(pathof(KomaMRI)), "../examples/2.phantoms/brain.h5")
  pog = grid.read_grid_phantom_jemris(brain_h5)
  return convertToDemoCDP(pog)
end

Lazy.@forward CurrentDensityPhantom.pog KomaMRI.plot_phantom_map
function plot_current_density(cdp::CurrentDensityPhantom)
  flat = grid.to_flat_phantom(cdp.pog)
  mask = cdp.pog.ρ .!= 0
  colour_indicator = sum([cdp.jx[mask] .^ 2, cdp.jy[mask] .^ 2, cdp.jz[mask] .^ 2])
  GLMakie.arrows(flat.x, flat.y, flat.z, cdp.jx[mask], cdp.jy[mask], cdp.jz[mask], axis=(type=GLMakie.Axis3,),
    arrowsize=0.0002, linewidth=0.00005, arrowcolor=colour_indicator, linecolor=colour_indicator)
end

function curl(a1, a2, a3, b1, b2, b3)
  return (
    a2 .* b3 - a3 .* b2,
    -(a1 .* b3 - a3 .* b1),
    a1 .* b2 - a2 .* b1
  )
end

function calculate_magnetic_field(cdp::CurrentDensityPhantom)::VectorField
  mu_0 = 12.0

  # s = max(grid.get_FOV(cdp.pog)...)
  Mx, My, Mz = size(cdp.pog.ρ)
  Mx_half, My_half, Mz_half = rtoi(Mx / 2), rtoi(My / 2), rtoi(Mz / 2)
  N = 2 * max(Mx, My, Mz)
  center = rtoi(N / 2)
  Mx_range = center-Mx_half:center+Mx_half-iseven(Mx)
  My_range = center-My_half:center+My_half-iseven(My)
  Mz_range = center-Mz_half:center+Mz_half-iseven(Mz)
  M_range = Mx_range, My_range, Mz_range

  k1 = fftfreq(N)
  g1 = zeros(ComplexF64, (N, N, N))
  g2 = zeros(ComplexF64, (N, N, N))
  g3 = zeros(ComplexF64, (N, N, N))
  for i in 1:size(cdp.pog.ρ, 1)
    for j in 1:size(cdp.pog.ρ, 2)
      for k in 1:size(cdp.pog.ρ, 3)
        kvec = [k1[i], k1[j], k1[k]]
        norm_sq = sum(kvec .^ 2)
        g1[i, j, k] = -1.0im * kvec[1] ./ norm_sq
        g2[i, j, k] = -1.0im * kvec[2] ./ norm_sq
        g3[i, j, k] = -1.0im * kvec[3] ./ norm_sq
      end
    end
  end
  g1[1] = 0
  g2[1] = 0
  g3[1] = 0

  padded_jx = zeros(Float64, (N, N, N))
  padded_jy = zeros(Float64, (N, N, N))
  padded_jz = zeros(Float64, (N, N, N))
  padded_jx[M_range...] = cdp.jx
  padded_jy[M_range...] = cdp.jy
  padded_jz[M_range...] = cdp.jz

  c1, c2, c3 = curl(fft(padded_jx), fft(padded_jy), fft(padded_jz), g1, g2, g3)
  B1, B2, B3 = real(ifft(c1)), real(ifft(c2)), real(ifft(c3))
  return mu_0 .* (B1[M_range...], B2[M_range...], B3[M_range...])
end

function plot_magnetic_field(cdp::CurrentDensityPhantom)
  flat = grid.to_flat_phantom(cdp.pog)
  mask = cdp.pog.ρ .!= 0
  B1, B2, B3 = calculate_magnetic_field(cdp)
  colour_indicator = sum([B1[mask] .^ 2, B2[mask] .^ 2, B3[mask] .^ 2])
  GLMakie.arrows(flat.x, flat.y, flat.z, B1[mask], B2[mask], B3[mask], axis=(type=GLMakie.Axis3,),
    arrowsize=0.0001, linewidth=0.00005, arrowcolor=colour_indicator, linecolor=colour_indicator)
end

function solve(Bz0::Array{Float64,3})::CurrentDensityPhantom
  f(x) = LinearAlgebra.norm(Bz - Bz0) .^ 2 / 2 + alpha / 2 * LinearAlgebra.norm(B) / σ + R(σ)
  x0 = []  # all B-field values in flat, along with sigma
  Optim.optimize(f, x0, Optim.LBFGS())
end

greet() = print("Hello World!")
end
