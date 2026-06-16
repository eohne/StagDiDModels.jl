* Benchmark Stata `did_imputation` (Borusyak et al.) on the SAME CSV panels that
* run_bench_julia.jl generated. Matched spec: all post horizons, 4 pre-trends,
* clustered on unit, minn(0), tol(1e-6), maxit(100), autosample.
*
* Stata SE is single-threaded, so this is the like-for-like comparison against
* our single-threaded estimator. Run from the bench/ directory.

clear all
set more off
version 17

* did_imputation needs reghdfe + ftools; confirm they're present.
capture which reghdfe
if _rc {
    di as error "reghdfe not installed -- run: ssc install reghdfe, ftools"
    exit 198
}

local cases S M L

* Warm up the ado / Mata libraries on a tiny run so they aren't timed below.
quietly {
    import delimited "data_S.csv", clear case(preserve)
    capture did_imputation Y id t Ei, allhorizons pretrends(4) cluster(id) ///
        minn(0) tol(0.000001) maxit(100) autosample
}

* Results accumulator.
postutil clear
tempname pf
postfile `pf' str4 case long nobs int ncoef double stata_s double b0 ///
    using "results_stata.dta", replace

foreach c of local cases {
    import delimited "data_`c'.csv", clear case(preserve)
    local nobs = _N

    * Two timed runs, keep the faster (reduces noise; no JIT so both comparable).
    local best = .
    forvalues r = 1/2 {
        timer clear 1
        timer on 1
        did_imputation Y id t Ei, allhorizons pretrends(4) cluster(id) ///
            minn(0) tol(0.000001) maxit(100) autosample
        timer off 1
        quietly timer list 1
        if (`best' == . | r(t1) < `best') local best = r(t1)
    }

    local ncoef = colsof(e(b))
    * Horizon-0 effect for cross-check against Julia's b(τ0).
    local b0 = .
    capture local b0 = _b[tau0]

    post `pf' ("`c'") (`nobs') (`ncoef') (`best') (`b0')
    di as txt "case `c': nobs=`nobs' ncoef=`ncoef' time=`best's b(tau0)=`b0'"
}

postclose `pf'
use "results_stata.dta", clear
export delimited using "results_stata.csv", replace
list, clean noobs
