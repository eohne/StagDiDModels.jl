"""
    BJSModel <: RegressionModel

Borusyak, Jaravel, and Spiess (2021) imputation estimator results.

"""
struct BJSModel <: RegressionModel
    β::AbstractVector                           # coefficients
    Σ::AbstractMatrix                           # vcov matrix  
    names::AbstractVector                       # coefficient names
    nobs::Int                       # number of observations
    dof_resid::Int                  # residual degrees of freedom
    
    # BJS-specific diagnostics
    y_name::Symbol                  # outcome variable name
    estimator_type::Symbol          # :static or :dynamic
    treatment_periods::Vector{Int}  # event times estimated (empty for static)
    first_stage_r2::Float64         # R² from Y(0) first stage regression
    n_treated::Int                  # number of treated observations
    n_donors::Int                   # number of donor observations
    control_type::Symbol            # :notyet or :never
    boot::Bool                      # true or false
    n_boot::Int                    # Bootstrap Iterations
    cluster::Symbol                 # Cluster var
end


# Basic interface for both models
for ModelType in [:BJSModel]
    @eval begin
        StatsAPI.responsename(m::$ModelType) = string(m.y_name)
        StatsAPI.coef(m::$ModelType) = m.β
        StatsAPI.vcov(m::$ModelType) = m.Σ
        StatsAPI.coefnames(m::$ModelType) = String.(m.names)
        StatsAPI.nobs(m::$ModelType) = m.nobs
        StatsAPI.dof_residual(m::$ModelType) = m.dof_resid
        StatsAPI.dof(m::$ModelType) = length(m.β)
        
        # Standard errors
        StatsAPI.stderror(m::$ModelType) = sqrt.(diag(StatsAPI.vcov(m)))
        
        # t-statistics (use StatsBase)
        tstatistic(m::$ModelType) = StatsAPI.coef(m) ./ StatsAPI.stderror(m)
        
        # p-values (two-tailed t-test)
        function StatsAPI.pvalue(m::$ModelType)
            ts = tstatistic(m)
            df = StatsAPI.dof_residual(m)
            return 2 .* ccdf.(Ref(TDist(df)), abs.(ts))
        end
        
        # Confidence intervals
        function StatsAPI.confint(m::$ModelType, level::Real=0.95)
            α = (1 - level) / 2
            ts = tstatistic(m) 
            df = StatsAPI.dof_residual(m)
            critical = quantile(TDist(df), 1 - α)
            se = StatsAPI.stderror(m)
            coefs = StatsAPI.coef(m)
            return hcat(coefs .- critical .* se, coefs .+ critical .* se)
        end
    end
end

# R2 - BJS uses first stage, Gardner uses second stage  
StatsAPI.r2(m::BJSModel) = m.first_stage_r2
StatsAPI.islinear(::BJSModel) = true



# BJS doesn't have meaningful adj R2
StatsAPI.adjr2(m::BJSModel) = NaN

# BJS doesn't have meaningful F-statistic (not a traditional regression)
fstatistic(m::BJSModel) = (NaN, 0, StatsAPI.dof_residual(m))

function StatsAPI.coeftable(m::BJSModel)
    β = StatsAPI.coef(m)
    se = StatsAPI.stderror(m) 
    ts = tstatistic(m)
    pvals = StatsAPI.pvalue(m)
    names = StatsAPI.coefnames(m)
    
    return CoefTable(
        hcat(β, se, ts, pvals),
        ["Coef.", "Std. Error", "t", "Pr(>|t|)"],
        names,
        4,  # p-value column for formatting
        3   # test statistic column  
    )
end

function top(m::BJSModel)
    out = [
            "Number of obs" sprint(show, StatsAPI.nobs(m), context = :compact => true);
            "Outcome" sprint(show, m.y_name, context = :compact => true);
            "Treated obs" sprint(show, m.n_treated, context = :compact => true);
            "Est. Type" sprint(show, m.estimator_type, context = :compact => true);
            "Donor obs" sprint(show, m.n_donors, context = :compact => true);
            "R² First Stage" @sprintf("%.3f",m.first_stage_r2);
            ]
    if m.boot
        out= vcat(out,[
            "Bootstrapped" sprint(show,m.boot,context = :compact => true);
            "N Boot" sprint(show, m.n_boot,context = :compact => true)]) 
    end
    out= vcat(out,[
        "Cluster" sprint(show, m.cluster,context = :compact => true)]) 

    return out
end

function Base.show(io::IO, m::BJSModel)
    ct = coeftable(m)
    #copied from show(iio,cf::Coeftable)
    cols = ct.cols; rownms = ct.rownms; colnms = ct.colnms;
    nc = length(cols)
    nr = length(cols[1])
    if length(rownms) == 0
        rownms = [lpad("[$i]",floor(Integer, log10(nr))+3) for i in 1:nr]
    end
    mat = [j == 1 ? NoQuote(rownms[i]) :
           j-1 == ct.pvalcol ? NoQuote(sprint(show, PValue(cols[j-1][i]))) :
           j-1 in ct.teststatcol ? StatsBase.TestStat(cols[j-1][i]) :
           cols[j-1][i] isa AbstractString ? NoQuote(cols[j-1][i]) : cols[j-1][i]
           for i in 1:nr, j in 1:nc+1]
    io = IOContext(io, :compact=>true, :limit=>false)
    A = Base.alignment(io, mat, 1:size(mat, 1), 1:size(mat, 2),
                       typemax(Int), typemax(Int), 3)
    nmswidths = pushfirst!(length.(colnms), 0)
    A = [nmswidths[i] > sum(A[i]) ? (A[i][1]+nmswidths[i]-sum(A[i]), A[i][2]) : A[i]
         for i in 1:length(A)]
    totwidth = sum(sum.(A)) + 2 * (length(A) - 1)


    #intert my stuff which requires totwidth
    ctitle = string(typeof(m))
    halfwidth = div(totwidth - length(ctitle), 2)
    print(io, " " ^ halfwidth * ctitle * " " ^ halfwidth)
    ctop = top(m)
    for i in 1:size(ctop, 1)
        ctop[i, 1] = ctop[i, 1] * ":"
    end
    println(io, '\n', repeat('=', totwidth))
    halfwidth = div(totwidth, 2) - 1
    interwidth = 2 +  mod(totwidth, 2)
    for i in 1:(div(size(ctop, 1) - 1, 2)+1)
        print(io, ctop[2*i-1, 1])
        print(io, lpad(ctop[2*i-1, 2], halfwidth - length(ctop[2*i-1, 1])))
        print(io, " " ^interwidth)
        if size(ctop, 1) >= 2*i
            print(io, ctop[2*i, 1])
            print(io, lpad(ctop[2*i, 2], halfwidth - length(ctop[2*i, 1])))
        end
        println(io)
    end
   
    # rest of coeftable code
    println(io, repeat('=', totwidth))
    print(io, repeat(' ', sum(A[1])))
    for j in 1:length(colnms)
        print(io, "  ", lpad(colnms[j], sum(A[j+1])))
    end
    println(io, '\n', repeat('─', totwidth))
    for i in 1:size(mat, 1)
        Base.print_matrix_row(io, mat, A, i, 1:size(mat, 2), "  ")
        i != size(mat, 1) && println(io)
    end
    println(io, '\n', repeat('=', totwidth))
    nothing
end

"""
    BJSModel(β, Σ, names, nobs, dof_resid; kwargs...)

Construct a BJSModel with required statistics and optional diagnostics.
"""
function BJSModel(β::AbstractVector, Σ::AbstractMatrix, names::AbstractVector, 
                  nobs::Int, dof_resid::Int;
                  y_name::Symbol = :y,
                  estimator_type::Symbol = :static,
                  treatment_periods::Vector{Int} = Int[],
                  first_stage_r2::Float64 = 0.0,
                  n_treated::Int = 0,
                  n_donors::Int = 0, 
                  control_type::Symbol = :notyet,
                  boot::Bool = false,
                  n_boot::Int = 0,
                  cluster::Symbol = :none)
    
    return BJSModel(β, Σ, names, nobs, dof_resid, y_name, estimator_type,
                    treatment_periods, first_stage_r2, n_treated, n_donors, control_type,boot,n_boot,cluster)
end