module BoseHubbardScalingThesisFigures
using Statistics, CairoMakie
const R=joinpath(@__DIR__,"results"); const OUT=joinpath(R,"thesis_figures")
parsev(s)=lowercase(s)=="true" ? true : lowercase(s)=="false" ? false : tryparse(Int,s)!==nothing ? parse(Int,s) : tryparse(Float64,s)!==nothing ? parse(Float64,s) : s
function readcsv(name)
 l=readlines(joinpath(R,name)); n=Symbol.(split(l[1],','))
 [NamedTuple{Tuple(n)}(Tuple(parsev.(split(x,',')))) for x in l[2:end] if !isempty(strip(x))]
end
function main()
 lazy=readcsv("bose_hubbard_lazy_dihedral_scaling.csv")
 l14=readcsv("bose_hubbard_linear_dihedral_L14.csv")
 println("lazy columns: ",join(string.(propertynames(lazy[1])),", "))
 println("L14 columns: ",join(string.(propertynames(l14[1])),", "))
 println("rows: lazy=",length(lazy)," L14=",length(l14))
 println("lazy L: ",sort(unique(x.L for x in lazy))," sym: ",sort(unique(String(x.symmetry) for x in lazy))," epochs: ",sort(unique(x.epoch for x in lazy)))
 println("L14 L: ",sort(unique(x.L for x in l14))," sym: ",sort(unique(String(x.symmetry) for x in l14))," epochs: ",sort(unique(x.epoch for x in l14)))
end
end
if abspath(PROGRAM_FILE)==@__FILE__; BoseHubbardScalingThesisFigures.main(); end
