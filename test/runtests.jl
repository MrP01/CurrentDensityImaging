using Test
import KomaMRI
import CurrentDensityImaging.GridPhantom as GP

@testset "Grid Phantom" begin
  brain_h5 = joinpath(dirname(pathof(KomaMRI)), "../examples/2.phantoms/brain.h5")
  pog = GP.read_grid_phantom_jemris(brain_h5)
  koma_phantom = KomaMRI.read_phantom_jemris(brain_h5)
  pog_phantom::KomaMRI.Phantom = GP.to_flat_phantom(pog)
  @test pog_phantom.x == koma_phantom.x
  @test pog_phantom.y == koma_phantom.y
  @test pog_phantom.z == koma_phantom.z
end
