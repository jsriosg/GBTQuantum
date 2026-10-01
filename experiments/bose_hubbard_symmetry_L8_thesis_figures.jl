module BoseHubbardSymmetryL8ThesisFigures
using Statistics, CairoMakie
const R=joinpath(@__DIR__,"results"); const OUT=joinpath(R,"thesis_figures")
parsev(s)=lowercase(s)=="true" ? true : lowercase(s)=="false" ? false : tryparse(Int,s)!==nothing ? parse(Int,s) : tryparse(Float64,s)!==nothing ? parse(Float64,s) : s
function readcsv(name)
 l=readlines(joinpath(R,name)); n=Symbol.(split(l[1],','))
 [NamedTuple{Tuple(n)}(Tuple(parsev.(split(x,',')))) for x in l[2:end] if !isempty(strip(x))]
end
function cols(rows); propertynames(rows[1]); end
function showcols(name,rows); println(name,": ",join(string.(cols(rows)),", ")); end
function main()
 trans=readcsv("bose_hubbard_translation_symmetry_L8.csv")
 dih=readcsv("bose_hubbard_fast_dihedral_L8.csv")
 cost=readcsv("bose_hubbard_dihedral_cost_grid_L8.csv")
 showcols("translation",trans); showcols("dihedral",dih); showcols("cost",cost)
 println("Rows: translation=",length(trans)," dihedral=",length(dih)," cost=",length(cost))
end
end
if abspath(PROGRAM_FILE)==@__FILE__; BoseHubbardSymmetryL8ThesisFigures.main(); end
