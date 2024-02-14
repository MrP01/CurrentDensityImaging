module grid
import KomaMRI
import HDF5

@kwdef struct PhantomOnAGrid
  name::String
  ρ::Array{Float64,3}
  T1::Array{Float64,3}
  T2::Array{Float64,3}
  T2s::Array{Float64,3}
  Δw::Array{Float64,3}
  Δx::Vector{Float64}
  offset::Vector{Float64}
end

function KomaMRI.plot_phantom_map(pog::PhantomOnAGrid, key::Symbol)
  KomaMRI.plot_phantom_map(to_flat_phantom(pog), key)
end

function read_grid_phantom_jemris(filename::String)
  fid = HDF5.h5open(filename)
  data = read(fid["sample/data"])
  return PhantomOnAGrid(
    name=basename(filename),
    ρ=data[1, :, :, :],
    T1=1e-3 ./ data[2, :, :, :],
    T2=1e-3 ./ data[3, :, :, :],
    T2s=1e-3 ./ data[4, :, :, :],
    Δw=data[5, :, :, :],
    Δx=vec(read(fid["sample/resolution"])) * 1e-3, # [m]
    offset=vec(read(fid["sample/offset"])) * 1e-3  # [m]
  )
end

function get_FOV(pog::PhantomOnAGrid)
  X, Y, Z = size(pog.ρ)
  FOVx = (X - 1) * pog.Δx[1] # [m]
  FOVy = (Y - 1) * pog.Δx[2] # [m]
  FOVz = (Z - 1) * pog.Δx[3] # [m]
  return FOVx, FOVy, FOVz
end

function to_flat_phantom(pog::PhantomOnAGrid)
  mask = pog.ρ .!= 0

  FOVx, FOVy, FOVz = get_FOV(pog)
  xx = reshape((-FOVx/2:pog.Δx[1]:FOVx/2), :, 1, 1)  #[(-FOVx/2:Δx[1]:FOVx/2)...;]
  yy = reshape((-FOVy/2:pog.Δx[2]:FOVy/2), 1, :, 1)  #[(-FOVy/2:Δx[2]:FOVy/2)...;;]
  zz = reshape((-FOVz/2:pog.Δx[3]:FOVz/2), 1, 1, :)  #[(-FOVz/2:Δx[3]:FOVz/2)...;;;]
  x = xx * 1 .+ yy * 0 .+ zz * 0 .+ pog.offset[1]  #spin x coordinates
  y = xx * 0 .+ yy * 1 .+ zz * 0 .+ pog.offset[2]  #spin y coordinates
  z = xx * 0 .+ yy * 0 .+ zz * 1 .+ pog.offset[3]  #spin z coordinates

  return KomaMRI.Phantom(
    name=pog.name,
    x=x[mask],
    y=y[mask],
    z=z[mask],
    ρ=pog.ρ[mask],
    T1=pog.T1[mask],
    T2=pog.T2[mask],
    T2s=pog.T2s[mask],
    Δw=pog.Δw[mask],
  )
end

export PhantomOnAGrid, to_flat_phantom
end
