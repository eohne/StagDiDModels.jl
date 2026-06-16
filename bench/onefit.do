* One did_imputation fit on data_L.csv, reporting Stata's own memory accounting.
* `memory` prints Stata's allocated memory (data + Mata + overhead) — the closest
* analogue to a working-set footprint. Process peak working set is captured
* separately by the PowerShell harness that launches this.
clear all
set more off
import delimited "data_L.csv", clear case(preserve)
* warmup so Mata libs are loaded and not counted as first-call growth
quietly did_imputation Y id t Ei, allhorizons pretrends(4) cluster(id) ///
    minn(0) tol(0.000001) maxit(100) autosample
di "===== memory after fit ====="
memory
