


#' Logistic regression
#'
blr <- function(
  formula, data, subset, na.action,
  nu = 1, mu0 = NULL, tau.sq = NULL, start = NULL,
  control = NULL, contrasts = NULL, model = TRUE
) {
  ## Check inputs
  ##  - nu
  if ( length(nu) > 1L )
    nu <- nu[1L]
  if ( nu < 1 )
    stop("Degrees of freedom parameter must be >= 1")
  ##  - control
  if ( is.null(control) )
    control <- bglm.control()
  else if ( class(control) != "bglm.control" )
    stop("control parameter must be specified through bglm.control")
  ##  - tau.sq
  if ( !is.null(tau.sq) && any(tau.sq <= 0) )
    stop("Prior scale tau.sq must be >= 0")
  ##
  ## Formula parsing idiom from lm()
  cl <- match.call()
  mf <- match.call( expand.dots = FALSE )
  m <- match(
    c("formula", "data", "subset", "na.action"), names(mf), 0L
  )
  mf <- mf[ c(1L, m) ]
  mf$drop.unused.levels <- TRUE
  mf[[ 1L ]] <- quote(stats::model.frame)
  mf <- eval( mf, parent.frame() )
  mt <- attr(mf, "terms")
  ##
  ## Extract x and y
  y <- model.response(mf, "numeric")
  x <- if ( !is.empty.model(mt) ) model.matrix(mt, mf, contrasts)
       else NULL
  n <- NROW(y)
  ##
  ## Initialize parameters if model is non-empty
  if ( !is.null(x) ) {
    ##
    ## Prior location mu0
    if ( is.null(mu0) )
      mu0 <- numeric(NCOL(x))
    else if ( length(mu0) == 1L )
      mu0 <- rep(mu0, NCOL(x))
    else if ( length(mu0) != NCOL(x) ) {
      msg <- sprintf(
        "Prior location mu0 given with length = %d, but should have length = %d",
        length(mu0), NCOL(x))
      stop(msg)
    }
    ##
    ## Prior variance tau^2
    if ( is.null(tau.sq) )
      tau.sq <- rep(control$tau.sq, NCOL(x))
    else if ( length(tau.sq) == 1L )
      tau.sq <- rep(tau.sq, NCOL(x))
    else if ( length(tau.sq) != NCOL(x) ) {
      msg <- sprintf(
        "Prior scale tau.sq given with length = %d, but should have length = %d",
        length(tau.sq), NCOL(x))
      stop(msg)
    }
    ##
    ## Starting values for beta
    if ( is.null(start) )
      start <- numeric(NCOL(x))
    if ( length(start) != NCOL(x) ) {
      msg <- sprintf(
        "Starting value given with length = %d, but should have length = %d",
        length(start), NCOL(x))
      stop(msg)
    }
  }
  res <- if ( is.null(x) )
           list(
             coefficients = numeric(), vcov = matrix(NA, 0, 0),
             ## deriv = numeric(),
             converged = TRUE, iter = 0,
             tau.sq = tau.sq, nu = nu, mu0 = mu0,
             residuals = y, fitted.values = 0 * y
           )
         else
           blr.fit(x, y, start, tau.sq, nu, mu0, control)
  class(res) <- "blr"
  res$model <- if (model) mf else data.frame()
  res$call <- cl
  res$terms <- mt
  res$contrasts <- attr(x, "contrasts")
  res$na.action <- attr(mf, "na.action")
  res
}


blr.fit <- function(x, y, theta, tau.sq, nu, mu0, control) {
  stopifnot(is(control, "bglm.control"))
  stopifnot(NROW(x) == length(y))
  stopifnot(NCOL(x) == length(theta))
  stopifnot(NCOL(x) == length(tau.sq))
  stopifnot(NCOL(x) == length(mu0))
  stopifnot(nu >= 1)
  stopifnot(all(tau.sq > 0))
  ##
  r <- .Call("fit_blr_cpp", x, y, theta, nu, tau.sq, mu0,
             control$maxit, control$xtol, control$xtol_rel,
             pacakge = "vb")
  names(r$coefficients) <- colnames(x)
  dimnames(r$vcov) <- list(colnames(x), colnames(x))
  r
}







## --- Methods for "brlm" --------------------------------------------

print.blr <- stats:::print.lm

vcov.blr <- function(object, ...) {
  object$vcov
}

model.matrix.blr <- function(object, ...) {
  model.matrix( object$terms, object$model )
}




logLik.blr <- function(object, ...) {
  y <- model.response(object$model)
  mu <- fitted(object)
  p <- length(object$coefficients)  # approximate
  n <- length(res)
  val <- y * log(mu) + (1 - y) * log1p(-mu)
  attr(val, "nall") <- n
  attr(val, "nobs") <- n - p
  attr(val, "df") <- p
  class(val) <- "logLik"
  val
}

deviance.blr <- function(object, ...) {
  fitted.llk <- as.numeric(logLik(object))
  -2 * fitted.llk
}






summary.blr <- function(object, ...) {
  res <- structure(
    list(
      call = object$call,
      terms = object$terms,
      cov.scaled = vcov(object),
      coefficiets = NULL,
      df = NULL,
      deviance = NULL,
      null.deviance = NULL,
      r.squared = NULL
    ),
    class = "summary.brlm"
  )
  se <- sqrt(diag(vcov(object)))
  res$coefficients <- cbind(
    "Estimate" = coef(object),
    "Std. Error" = se,
    "t value" = coef(object) / se
  )
  rownames(res$coefficients) <- names(coef(object))
  ##
  ## Deviance
  p <- length(object$coefficients)  # approximate
  n <- nrow(object$model)
  res$df <- c( p, n - p )
  ## y <- model.response(object$model)  # Null model
  ## nu <- object$nu
  ## fit0 <- brlm(y ~ 1, nu = nu)
  res$deviance <- deviance(object)
  ## res$null.deviance <- deviance(fit0)
  ##
  res
}


## print.summary.blr <- function(x, digits = 3L, ...) {
##   cat("\nCall:\n", paste(deparse(x$call), sep = "\n", collapse = "\n"),
##       "\n\n", sep = "")
##   cat("Residuals:\n")
##   rq <- structure(zapsmall(quantile(x$residuals)),
##                   names = c("Min", "1Q", "Median", "3Q", "Max"))
##   print(rq, digits = digits)
##   cat("\n")
##   printCoefmat(x$coefficients, digits = digits)
##   cat("\nResidual standard error:",
##       formatC(x$sigma, digits = digits),
##       "on", x$df[2L], "degrees of freedom\n")
##   cat("R-squared:", formatC(x$r.squared, digits = digits), "\n")
##   invisible(x)
## }




