module CurrentDensityImaging
import KomaMRI
import Lazy
import LinearAlgebra
import FastTransforms: fft, ifft, fftfreq
import GLMakie
# import CairoMakie
import Optim
include("./grid.jl")

const μ_0 = 12.0
const FieldComponent = Array{Float64,3}
const VectorField = Tuple{FieldComponent,FieldComponent,FieldComponent}

@kwdef struct CurrentDensityPhantom
  pog::grid.PhantomOnAGrid  # phantom on grid
  jx::FieldComponent = ones(size(pog.ρ))
  jy::FieldComponent = zeros(size(pog.ρ))
  jz::FieldComponent = zeros(size(pog.ρ))
end

rtoi(x) = Int32(round(x))

function generateDemoPOG(shape=(8, 8, 4))::grid.PhantomOnAGrid
  sero = zeros(Float64, shape)
  return grid.PhantomOnAGrid(name="demo phantom!", ρ=ones(shape), T1=sero, T2=sero, T2s=sero, Δw=sero, Δx=vec([1.0, 1.0, 1.0]), offset=vec([0, 0, 0]))
end
function convertToHomoCDP(pog::grid.PhantomOnAGrid)::CurrentDensityPhantom
  shape = size(pog.ρ)
  jz = zeros(Float64, shape)
  jz[rtoi(shape[1] / 3):rtoi(shape[1] * 2 / 3), rtoi(shape[2] / 3):rtoi(shape[2] * 2 / 3), :] .= 0.2
  jz .*= 1:shape[1]
  return CurrentDensityPhantom(pog, zeros(Float64, shape), zeros(Float64, shape), jz)
end
function convertToDemoCDP(pog::grid.PhantomOnAGrid)::CurrentDensityPhantom
  shape = size(pog.ρ)
  jx = zeros(Float64, shape)
  jy = zeros(Float64, shape)
  jz = zeros(Float64, shape)
  jx[rtoi(shape[1] / 3):rtoi(shape[1] * 2 / 3), rtoi(shape[2] / 3):rtoi(shape[2] * 2 / 3), :] .= -0.3
  jy[rtoi(shape[1] / 3):rtoi(shape[1] * 2 / 3), rtoi(shape[2] / 3):rtoi(shape[2] * 2 / 3), :] .= -0.3
  jz[rtoi(shape[1] / 3):rtoi(shape[1] * 2 / 3), rtoi(shape[2] / 3):rtoi(shape[2] * 2 / 3), :] .= 0.4
  jz .*= 1:shape[1]
  return CurrentDensityPhantom(pog, jx, jy, jz)
end
function generateDemoCDP(shape=(8, 8, 4))::CurrentDensityPhantom
  return convertToDemoCDP(generateDemoPOG(shape))
end
function generateHomoCDP(shape=(8, 8, 4))::CurrentDensityPhantom
  return convertToHomoCDP(generateDemoPOG(shape))
end
function loadBrainCDP()::CurrentDensityPhantom
  brain_h5 = joinpath(dirname(pathof(KomaMRI)), "../examples/2.phantoms/brain.h5")
  pog = grid.read_grid_phantom_jemris(brain_h5)
  return convertToDemoCDP(pog)
end

Lazy.@forward CurrentDensityPhantom.pog KomaMRI.plot_phantom_map
function plot_current_density(cdp::CurrentDensityPhantom; backend=GLMakie, factor=1.0)
  flat = grid.to_flat_phantom(cdp.pog)
  # mask = (cdp.pog.ρ .!= 0) .& ((cdp.jx .^ 2 + cdp.jy .^ 2 + cdp.jz .^ 2) .> 1e-9)
  mask = (cdp.pog.ρ .!= 0) .& ((cdp.jx .^ 2 + cdp.jy .^ 2 + cdp.jz .^ 2) .> 0.0)
  colour_indicator = sum([cdp.jx[mask] .^ 2, cdp.jy[mask] .^ 2, cdp.jz[mask] .^ 2])
  fig = backend.Figure()
  ax = backend.Axis3(fig[1, 1])
  backend.arrows!(flat.x[mask[:]], flat.y[mask[:]], flat.z[mask[:]],
    cdp.jx[mask] * factor, cdp.jy[mask] * factor, cdp.jz[mask] * factor,
    arrowcolor=colour_indicator, linecolor=colour_indicator)
  return fig
end

cross(a1, a2, a3, b1, b2, b3) = (
  a2 .* b3 - a3 .* b2,
  -(a1 .* b3 - a3 .* b1),
  a1 .* b2 - a2 .* b1
)

function centraldiff(X; dims)
  if dims == 1
    A = X[1:end-1, :, :]
    B = X[2:end, :, :]
  end
  if dims == 2
    A = X[:, 1:end-1, :]
    B = X[:, 2:end, :]
  end
  if dims == 3
    A = X[:, :, 1:end-1]
    B = X[:, :, 2:end]
  end
  return (diff(A, dims=dims) + diff(B, dims=dims)) ./ 2
end

function curl(B1::FieldComponent, B2::FieldComponent, B3::FieldComponent)::VectorField
  Mx, My, Mz = size(B1)
  jx, jy, jz = zeros(Mx, My, Mz), zeros(Mx, My, Mz), zeros(Mx, My, Mz)
  jx[:, 2:My-1, 2:Mz-1] = centraldiff(B3, dims=2)[:, :, 2:Mz-1] - centraldiff(B2, dims=3)[:, 2:My-1, :]
  jy[2:Mx-1, :, 2:Mz-1] = centraldiff(B1, dims=3)[2:Mx-1, :, :] - centraldiff(B3, dims=1)[:, :, 2:Mz-1]
  jz[2:Mx-1, 2:My-1, :] = centraldiff(B2, dims=1)[:, 2:My-1, :] - centraldiff(B1, dims=2)[2:Mx-1, :, :]
  return (jx, jy, jz)
end

function calculate_magnetic_field(cdp::CurrentDensityPhantom)::VectorField
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

  c1, c2, c3 = cross(fft(padded_jx), fft(padded_jy), fft(padded_jz), g1, g2, g3)
  B1, B2, B3 = real(ifft(c1)), real(ifft(c2)), real(ifft(c3))
  return μ_0 .* (B1[M_range...], B2[M_range...], B3[M_range...])
end

function reconstructCDPFromB(B1::FieldComponent, B2::FieldComponent, B3::FieldComponent)
  B_shape = size(B1)
  jx, jy, jz = curl(B1, B2, B3) ./ μ_0
  pog = generateDemoPOG(B_shape)
  return CurrentDensityPhantom(pog, jx, jy, jz)
end

function plot_magnetic_field(cdp::CurrentDensityPhantom; backend=GLMakie, factor=1.0)
  flat = grid.to_flat_phantom(cdp.pog)
  mask = cdp.pog.ρ .!= 0
  B1, B2, B3 = calculate_magnetic_field(cdp)
  colour_indicator = sum([B1[mask] .^ 2, B2[mask] .^ 2, B3[mask] .^ 2])
  fig = backend.Figure()
  ax = backend.Axis3(fig[1, 1])
  backend.arrows!(flat.x, flat.y, flat.z, B1[mask] * factor, B2[mask] * factor, B3[mask] * factor,
    arrowcolor=colour_indicator, linecolor=colour_indicator)
  return fig
end

function plot_conductivity(cdp::CurrentDensityPhantom, σ::FieldComponent; backend=GLMakie)
  flat = grid.to_flat_phantom(cdp.pog)
  mask = cdp.pog.ρ .!= 0
  fig = backend.Figure()
  ax = backend.Axis3(fig[1, 1], azimuth=0.3 * pi, elevation=0.06 * pi)
  backend.scatter!(flat.x, flat.y, flat.z, color=σ[mask], markersize=20)
  return fig
end

function to_flat(B1, B2, B3, σ; B_flat_size)::Vector{Float64}
  return [
    reshape(B1, (B_flat_size,))...,  # Bx
    reshape(B2, (B_flat_size,))...,  # By
    reshape(B3, (B_flat_size,))...,  # Bz
    reshape(σ, (B_flat_size,))...  # σ
  ]  # all B-field values in flat, along with sigma
end

function from_flat(x::Vector{Float64}; B_shape, B_flat_size)
  return (
    reshape(x[1:B_flat_size], B_shape),  # Bx
    reshape(x[B_flat_size+1:2*B_flat_size], B_shape),  # By
    reshape(x[2*B_flat_size+1:3*B_flat_size], B_shape),  # Bz
    reshape(x[3*B_flat_size+1:4*B_flat_size], B_shape)  # σ
  )
end

function objective(Bx, By, Bz, σ; Bz0)
  jx, jy, jz = curl(Bx, By, Bz)
  bz_match = 1e3 * sum((Bz - Bz0) .^ 2) / 2
  power_dissipation = 1e-4 / 2 * sum((jx .^ 2 + jy .^ 2 + jz .^ 2) ./ σ)  # in units of power (Watt)
  dBx, dBy, dBz = centraldiff(Bx, dims=1), centraldiff(By, dims=2), centraldiff(Bz, dims=3)
  divergence_penalty = sum((dBx[:, 2:end-1, 2:end-1] + dBy[2:end-1, :, 2:end-1] + dBz[2:end-1, 2:end-1, :]) .^ 2)
  σ_tv_regulariser = 1e-3 * (sum(abs.(centraldiff(σ, dims=1))) + sum(abs.(centraldiff(σ, dims=2))) + sum(abs.(centraldiff(σ, dims=3))))
  # @show bz_match, power_dissipation, divergence_penalty, σ_tv_regulariser
  return bz_match + power_dissipation + divergence_penalty + σ_tv_regulariser
end

function objectiveForCDP(cdp::CurrentDensityPhantom)
  Bx, By, Bz = calculate_magnetic_field(cdp)
  return objective(Bx, By, Bz, ones(size(Bx)); Bz0=Bz)
end

function find_matching_B(Bz0::FieldComponent)
  B_shape = size(Bz0)
  B_flat_size = prod(B_shape)
  x0 = to_flat(ones(B_shape), ones(B_shape), Bz0, ones(B_shape); B_flat_size)
  function f(x::Vector{Float64})
    Bx, By, Bz, σ = from_flat(x; B_shape, B_flat_size)
    return objective(Bx, By, Bz, σ; Bz0)
  end
  result = Optim.optimize(f, x0, method=Optim.LBFGS(), iterations=40)
  return result
end

function solve(Bz0::FieldComponent)::Tuple{CurrentDensityPhantom,FieldComponent}
  result = find_matching_B(Bz0)
  @show result
  B_shape = size(Bz0)
  B_flat_size = prod(B_shape)
  B1, B2, B3, σ = CDI.from_flat(result.minimizer; B_shape, B_flat_size)
  return reconstructCDPFromB(B1, B2, B3), σ
end
end
