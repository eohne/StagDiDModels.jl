suppressMessages({library(did2s); library(fixest); library(data.table)})
df <- fread("gpanel.csv")
df[, g := fifelse(is.na(Ei), 0L, as.integer(Ei))]
df[, rel := fifelse(g==0L, 999L, t-g)]
df[, treat := as.integer(g>0L & t>=g)]
m <- did2s(df, yname="Y", first_stage=~0|id+t,
           second_stage=~i(rel, ref=c(-1,999)), treatment="treat",
           cluster_var="id", verbose=FALSE)
co <- coef(m); se <- sqrt(diag(vcov(m)))
rel <- as.integer(sub("rel::", "", names(co), fixed=TRUE))
r <- data.table(rel=rel, coef_r=as.numeric(co), se_r=as.numeric(se))
j <- fread("gardner_julia_se.csv")
z <- merge(j, r, by="rel")[order(rel)]
z[, dcoef := abs(coef-coef_r)][, dse := abs(se-se_r)][, rse := dse/pmax(abs(se_r),1e-12)]
print(z)
cat(sprintf("\nmatched %d periods | max|Δcoef|=%.2e  max|Δse|=%.2e  max rel-se=%.2e\n",
            nrow(z), max(z$dcoef), max(z$dse), max(z$rse)))
