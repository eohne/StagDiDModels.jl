"""
    joint_pretrend_test(m::StatsAPI.StatisticalModel; 
                        leads::Vector{Int} = Int[-2, -3, -4],
                        pattern::Regex = r"τ::(-?\\d+)")

Joint χ² test that pre-treatment leads are zero: H₀: β_τ = 0 for τ ∈ leads.

Works with all DiD model types: `BJSModel`, `GardnerModel`, `SunabModel`, `TWFEModel`.

# Arguments
- `m::StatisticalModel`: Event study model with τ coefficients
- `leads::Vector{Int}`: Pre-treatment periods to test (default: [-2, -3, -4])
- `pattern::Regex`: Pattern to extract τ values (default matches "τ::-2", "τ::3", etc.)

# Returns
NamedTuple with:
- `chisq`: Chi-squared test statistic
- `df`: Degrees of freedom 
- `p`: P-value
- `tested`: Vector of τ values actually tested (subset of `leads` that were found)

# Examples
```julia
# Test standard pretrends for any estimator
m = fit_bjs_dynamic(df; y=:ret, id=:fund, t=:month, g=:g, pretrends=true)
pretest = joint_pretrend_test(m; leads=[-3, -2, -1])

# Test different leads
joint_pretrend_test(m; leads=[-5, -4, -3])

# Works with any estimator
m_sunab = fit_sunab(df; y=:ret, id=:fund, t=:month, g=:g)
joint_pretrend_test(m_sunab; leads=[-2, -1])
```
"""
function joint_pretrend_test(m::StatsAPI.StatisticalModel; 
                             leads::Vector{Int} = Int[-2, -3, -4],
                             pattern::Regex = r"τ::(-?\d+)")
    
    nms = StatsAPI.coefnames(m)
    β = StatsAPI.coef(m)
    Σ = StatsAPI.vcov(m)

    # Map τ -> index in coefficient vector
    τ_to_idx = Dict{Int,Int}()
    for (i, nm) in enumerate(nms)
        mm = match(pattern, nm)
        if mm !== nothing
            τ = parse(Int, mm.captures[end])
            τ_to_idx[τ] = i
        end
    end

    # Find which leads are actually available in the model
    available_leads = [ℓ for ℓ in leads if haskey(τ_to_idx, ℓ)]
    
    if isempty(available_leads)
        return (chisq=NaN, df=0, p=NaN, tested=Int[])
    end
    
    # Get coefficient indices for available leads
    sel = [τ_to_idx[ℓ] for ℓ in available_leads]

    # Extract relevant coefficients and covariance matrix
    b = β[sel]
    V = Σ[sel, sel]
    
    # Wald statistic: b' V⁻¹ b, with stable fallback if V not positive definite
    stat = try
        F = cholesky(Symmetric(V); check=false)
        if isposdef(F)
            x = F \ b     # solves V * x = b
            dot(b, x)     # b' V⁻¹ b
        else
            dot(b, pinv(Symmetric(V)) * b)
        end
    catch
        dot(b, pinv(Symmetric(V)) * b)
    end
    
    df = length(sel)
    p = ccdf(Chisq(df), stat)

    return (chisq=stat, df=df, p=p, tested=available_leads)
end


"""
    pre_trends_summary(m::StatsAPI.StatisticalModel; 
                            leads::Vector{Int} = Int[-5, -4, -3, -2],
                            α::Float64 = 0.05)

Comprehensive parallel trends diagnostics for any DiD estimator.

# Arguments
- `m::StatisticalModel`: Event study model
- `leads::Vector{Int}`: Pre-treatment periods to examine
- `α::Float64`: Significance level (default: 0.05)

# Returns
Detailed summary of parallel trends tests and individual coefficient significance.

# Example
```julia
m = fit_bjs_dynamic(df; y=:ret, id=:fund, t=:month, g=:g, pretrends=true)
pre_trends_summary(m)
```
"""
function pre_trends_summary(m::StatsAPI.StatisticalModel; 
                                 leads::Vector{Int} = Int[-5, -4, -3, -2],
                                 α::Float64 = 0.05)
    
    # Get joint test
    joint_test = joint_pretrend_test(m; leads=leads)
    
    # Get individual coefficients and p-values
    nms = StatsAPI.coefnames(m)
    coefs = StatsAPI.coef(m)
    pvals = StatsAPI.pvalue(m)
    
    # Extract pretrend coefficients
    pattern = r"τ::(-?\d+)"
    pretrend_info = []
    
    for (i, nm) in enumerate(nms)
        mm = match(pattern, nm)
        if mm !== nothing
            τ = parse(Int, mm.captures[end])
            if τ in leads && τ < 0  # Only negative τ (pre-treatment)
                push!(pretrend_info, (τ=τ, coef=coefs[i], pval=pvals[i], significant=pvals[i] < α))
            end
        end
    end
    
    # Sort by τ
    sort!(pretrend_info, by=x->x.τ)
    
    println("=" ^60)
    println("PRETREND TESTS")
    println("=" ^60)
    
    # Joint test results
    if joint_test.df > 0
        joint_result = joint_test.p > α ? "PASS" : "FAIL"
        println("Joint Test (H₀: all pretrends = 0)")
        println("  χ² = $(round(joint_test.chisq, digits=3)), df = $(joint_test.df)")
        println("  p-value = $(round(joint_test.p, digits=4)) $(joint_result)")
        println()
    else
        println(" No pretrend coefficients found for joint test")
        println()
    end
    
    # Individual coefficients
    if !isempty(pretrend_info)
        println("Individual Pretrend Coefficients:")
        println("  τ        Coefficient    P-value     Significant")
        println("  " * "-"^45)
        
        for info in pretrend_info
            sig_marker = info.significant ? "Y" : "N"
            println("  $(lpad(info.τ, 3))      $(rpad(round(info.coef, digits=4), 10))    $(rpad(round(info.pval, digits=4), 8))    $(sig_marker)")
        end
        
        n_significant = sum(x.significant for x in pretrend_info)
        println()
        println("Summary: $(n_significant)/$(length(pretrend_info)) individual pretrends significant at α = $(α)")
    else
        println(" No individual pretrend coefficients found")
    end
    
    println("=" ^60)
    
    # Overall assessment
    overall_pass = (joint_test.df > 0 && joint_test.p > α) && 
                   (isempty(pretrend_info) || sum(x.significant for x in pretrend_info) == 0)
    
    println("=" ^60)
    
    return (
        joint_test = joint_test,
        individual_tests = pretrend_info,
        overall_pass = overall_pass
    )
end