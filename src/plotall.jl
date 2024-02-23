include("./CurrentDensityImaging.jl")
import CairoMakie

const RESULTS_FOLDER = joinpath(@__DIR__, "..", "figures")

function save_fig(fig, name::String; throughEps=false)
  # name = _saveFigCommon(name)
  if isnothing(name)
    return
  end
  path = joinpath(RESULTS_FOLDER, name)
  # if ~isdir(joinpath(RESULTS_FOLDER, p.name))
  #   mkdir(joinpath(RESULTS_FOLDER, p.name))
  # end
  if throughEps
    CairoMakie.save("$path.eps", fig)
    run(`epstopdf $path.eps -o $path.pdf`)
  else
    CairoMakie.save("$path.pdf", fig)
  end
  @info "Exported $name.pdf"
end

function plot_all()
  cdp = CDI.generateDemoCDP((16, 16, 8))
  fig = CDI.plot_current_density(cdp; backend=CairoMakie)
  save_fig(fig, "demo-cdp-j-field"; throughEps=true)
  fig = CDI.plot_magnetic_field(cdp; backend=CairoMakie)
  save_fig(fig, "demo-cdp-b-field"; throughEps=true)
end
