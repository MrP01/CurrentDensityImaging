include("./CurrentDensityImaging.jl");
CDI = CurrentDensityImaging;
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

function plot_demo_cdp()
  cdp = CDI.generateDemoCDP()
  fig = CDI.plot_current_density(cdp; backend=CairoMakie)
  save_fig(fig, "demo-cdp-j-field"; throughEps=true)
  fig = CDI.plot_magnetic_field(cdp; backend=CairoMakie, factor=0.1)
  save_fig(fig, "demo-cdp-b-field"; throughEps=true)
end

function plot_homo_cdp()
  cdp = CDI.generateHomoCDP()
  fig = CDI.plot_current_density(cdp; backend=CairoMakie)
  save_fig(fig, "homo-cdp-j-field"; throughEps=true)
  fig = CDI.plot_magnetic_field(cdp; backend=CairoMakie, factor=0.1)
  save_fig(fig, "homo-cdp-b-field"; throughEps=true)
  B1, B2, B3 = CDI.calculate_magnetic_field(cdp)
  cdpbr = CDI.reconstructCDPFromB(B1, B2, B3)
  fig = CDI.plot_magnetic_field(cdpbr; backend=CairoMakie, factor=0.1)
  save_fig(fig, "homo-cdpbr-b-field"; throughEps=true)
end

function plot_reconstructions()
  cdp = CDI.generateDemoCDP()
  B1, B2, B3 = CDI.calculate_magnetic_field(cdp)
  cdpr, σr = CDI.solve(B3)
  B1r, B2r, B3r = CDI.calculate_magnetic_field(cdpr)
  fig = CDI.plot_current_density(cdpr; backend=CairoMakie, factor=4.0)
  save_fig(fig, "cdpr-j-field"; throughEps=true)
  fig = CDI.plot_magnetic_field(cdpr; backend=CairoMakie, factor=0.8)
  save_fig(fig, "cdpr-b-field"; throughEps=true)
  fig = CDI.plot_conductivity(cdpr, σr; backend=CairoMakie)
  save_fig(fig, "cdpr-sigma"; throughEps=true)

  cdprr, σrr = CDI.solve(B3r)
  # σrr = max.(σrr, 1.0)
  fig = CDI.plot_current_density(cdprr; backend=CairoMakie, factor=4.0)
  save_fig(fig, "cdprr-j-field"; throughEps=true)
  fig = CDI.plot_magnetic_field(cdprr; backend=CairoMakie, factor=0.8)
  save_fig(fig, "cdprr-b-field"; throughEps=true)
  fig = CDI.plot_conductivity(cdprr, σrr; backend=CairoMakie)
  save_fig(fig, "cdprr-sigma"; throughEps=true)
end

function plot_all()
  CairoMakie.activate!()
  # plot_demo_cdp()
  plot_homo_cdp()
  plot_reconstructions()
end
