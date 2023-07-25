
bglm.control <- function(
  maxit = 100L,
  tau.sq = 1e4,
  xtol = 1e-5,
  xtol_rel = 1e-10,
  ...
) {
  L <- list(...)
  if ( length(L) ) {
    warning("Unused parameters: ", paste(names(L), sep = ", "))
  }
  maxit <- abs(maxit[1L])
  tau.sq <- abs(tau.sq[1L])
  xtol <- abs(xtol[1L])
  xtol_rel <- abs(xtol_rel[1L])
  structure(
    list(maxit = maxit, tau.sq = tau.sq, xtol = xtol,
         xtol_rel = xtol_rel),
    class = "bglm.control"
  )
}

