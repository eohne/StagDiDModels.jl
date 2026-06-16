# Time the R equivalents of Gardner / Sun-Abraham / TWFE on the SAME data_*.csv
# panels Julia uses, with matched dynamic event-study specs clustered on unit.
# Warm timings (run once, then min of 3). Reports horizon-0 effect for parity.
suppressMessages({library(did2s); library(fixest); library(data.table)})

bestof <- function(f, reps = 3) {
  f()                                           # warm up (f is a thunk, re-evaluated each call)
  min(replicate(reps, system.time(f())[["elapsed"]]))
}
b0 <- function(m) {                             # horizon-0 (event time 0) coef
  co <- coef(m); idx <- grep("::0$", names(co))
  if (length(idx) >= 1) unname(co[idx[1]]) else NA_real_
}

cases <- list(S = "data_S.csv", M = "data_M.csv", L = "data_L.csv")
out <- data.frame()

for (cs in names(cases)) {
  df <- fread(cases[[cs]])
  df[, g    := fifelse(is.na(Ei), 0L, as.integer(Ei))]
  df[, rel  := fifelse(g == 0L, 999L, t - g)]               # never-treated sentinel
  df[, treat := as.integer(g > 0L & t >= g)]
  df[, gc   := fifelse(g == 0L, 10000L, g)]                 # sunab never-treated cohort
  n <- nrow(df)

  # Gardner two-stage (did2s)
  fg <- function() did2s(df, yname = "Y",
                         first_stage  = ~ 0 | id + t,
                         second_stage = ~ i(rel, ref = c(-1, 999)),
                         treatment = "treat", cluster_var = "id",
                         verbose = FALSE)
  tg <- bestof(fg); mg <- fg()

  # Sun-Abraham (fixest::sunab)
  fs <- function() feols(Y ~ sunab(gc, t) | id + t, data = df, cluster = ~ id)
  ts <- bestof(fs); ms <- fs()

  # Dynamic TWFE (fixest event study)
  ft <- function() feols(Y ~ i(rel, ref = c(-1, 999)) | id + t, data = df, cluster = ~ id)
  tt <- bestof(ft); mt <- ft()

  out <- rbind(out, data.frame(
    estimator = c("gardner", "sunab", "twfe"),
    case = cs, nobs = n,
    r_s = c(tg, ts, tt),
    b0  = c(b0(mg), b0(ms), b0(mt))))
  cat(sprintf("%-4s n=%-7d gardner=%.4fs(b0=%.4f)  sunab=%.4fs(b0=%.4f)  twfe=%.4fs(b0=%.4f)\n",
              cs, n, tg, b0(mg), ts, b0(ms), tt, b0(mt)))
}
fwrite(out, "results_estimators_r.csv")
cat("Wrote results_estimators_r.csv\n")
