using StagDiDModels, DataFrames, CSV, Random
rng=MersenneTwister(11)
N=300;T=10;cohorts=(4,6,8)
ids=Int[];ts=Int[];Ei=Union{Int,Missing}[];Y=Float64[]
α=randn(rng,N);γ=randn(rng,T)
for u in 1:N
  ei=(u%4==0) ? missing : cohorts[(u%3)+1]
  for tt in 1:T
    tr=!ismissing(ei)&&tt>=ei; k=ismissing(ei) ? 0 : tt-ei
    push!(ids,u);push!(ts,tt);push!(Ei,ei);push!(Y,α[u]+γ[tt]+(tr ? 0.3*(k+1) : 0.0)+0.4randn(rng))
  end
end
df=DataFrame(id=ids,t=ts,Ei=Ei,Y=Y); CSV.write("gpanel.csv",df)
m=fit_gardner_dynamic(df;y=:Y,id=:id,t=:t,g=:Ei,cluster=:id); se=stderror(m)
rows=NamedTuple[]
for (i,nm) in enumerate(coefnames(m))
  mm=match(r"::(-?\d+)$",nm); mm===nothing && continue
  push!(rows,(rel=parse(Int,mm.captures[1]),coef=coef(m)[i],se=se[i]))
end
CSV.write("gardner_julia_se.csv",DataFrame(rows)); println("julia k=",length(rows))
