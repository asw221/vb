


#' Robust linear regression
#'
brlm <- function(
  formula, data, subset, na.action,
  nu = 4, mu0 = NULL, tau.sq = NULL, start = NULL,
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
    ## Prior mean mu0
    if ( is.null(mu0) )
      mu0 <- numeric(NCOL(x))
    else if ( length(mu0) == 1L )
      mu0 <- rep(mu0, NCOL(x))
    else if ( length(mu0) != NCOL(x) ) {
      msg <- sprintf(
        "Prior mean mu0 given with length = %d, but should have length = %d",
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
        "Prior variance tau.sq given with length = %d, but should have length = %d",
        length(tau.sq), NCOL(x))
      stop(msg)
    }
    ##
    ## Starting values for beta
    if ( is.null(start) )
      start <- if ( NCOL(x) <= NROW(x) ) qr.solve(x, y)
               else numeric(NCOL(x))
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
             sigma = numeric(), weights = rep(1, length(y)),
             ## deriv = numeric(),
             converged = TRUE, iter = 0,
             tau.sq = tau.sq, nu = nu, mu0 = mu0,
             residuals = y, fitted.values = 0 * y
           )
         else
           brlm.fit(x, y, start, tau.sq, nu, mu0, control)
  class(res) <- "brlm"
  res$model <- if (model) mf else data.frame()
  res$call <- cl
  res$terms <- mt
  res$contrasts <- attr(x, "contrasts")
  res$na.action <- attr(mf, "na.action")
  res
}


brlm.fit <- function(x, y, theta, tau.sq, nu, mu0, control) {
  stopifnot(is(control, "bglm.control"))
  stopifnot(NROW(x) == length(y))
  stopifnot(NCOL(x) == length(theta))
  stopifnot(NCOL(x) == length(tau.sq))
  stopifnot(NCOL(x) == length(mu0))
  stopifnot(nu >= 1)
  stopifnot(all(tau.sq > 0))
  ##
  r <- .Call("fit_brlm_cpp", x, y, theta, nu, tau.sq, mu0,
             control$maxit, control$xtol, control$xtol_rel,
             pacakge = "vb")
  names(r$coefficients) <- colnames(x)
  dimnames(r$vcov) <- list(colnames(x), colnames(x))
  r
}


brlm.fit0 <- function(x, y, theta, tau.sq, nu, mu0, control) {
  sigma.sq <- var(c( y - x %*% theta ))
  converged <- FALSE
  i <- 0L
  while ( i < control$maxit && !converged ) {
    d <- brlm.deriv(theta, y, x, sigma.sq, tau.sq, nu, mu0)
    delta <- c( solve(d$hess) %*% d$deriv )
    theta <- theta - delta
    sigma.sq <- var(c( y - x %*% theta ))
    i <- i + 1L
    converged <- sum( delta^2 ) <= control$xtol
  }
  d <- brlm.deriv(theta, y, x, sigma.sq, tau.sq, nu, mu0)
  eta <- c( x %*% theta )
  list(
    coefficients = theta,
    vcov = -solve(d$hess),
    sigma = sqrt(sigma.sq),
    weights = d$weights,
    ## deriv = d$deriv,
    converged = converged,
    iter = i,
    tau.sq = tau.sq,
    nu = nu,
    mu0 = mu0,
    residuals = y - eta,
    fitted.values = eta
  )
}






brlm.deriv <- function(theta, y, x, sigma.sq, tau.sq, nu, mu0) {
  p <- NCOL(x)
  r <- c( y - x %*% theta )
  w <- 1 / (1 + r^2 / (nu * sigma.sq))
  scl <- (nu + 1) / (nu * sigma.sq)
  db <- scl * c( t(x) %*% (r * w) ) +  # loglik
    -(theta - mu0) / (sigma.sq * tau.sq)   # prior
  ## Compute Hessian
  dw <- 2 / (nu * sigma.sq) * r^2 * w^2
  h <- -scl * ( t(x * (w + dw)) %*% x )         # loglik
  diag(h) <- diag(h) - 1 / (sigma.sq * tau.sq)  # prior
  list( deriv = db, hess = h, weights = w )
}





## --- Methods for "brlm" --------------------------------------------

loocv <- function(object, ...) UseMethod( "loocv" )

print.brlm <- stats:::print.lm

vcov.brlm <- function(object, ...) {
  object$vcov
}

sigma.brlm <- function(object, ...) {
  object$sigma
}

logLik.brlm <- function(object, ...) {
  res <- residuals(object) / sigma(object)
  p <- length(object$coefficients)  # approximate
  n <- length(res)
  val <- sum(dt(res, object$nu, log = TRUE)) - n * log(sigma(object))
  attr(val, "nall") <- n
  attr(val, "nobs") <- n - p
  attr(val, "df") <- p
  class(val) <- "logLik"
  val
}

deviance.brlm <- function(object, ...) {
  n <- nrow(object$model)
  saturated.llk <- n * dt(0, object$nu, log = TRUE) -
    n * log(sigma(object))
  fitted.llk <- as.numeric(logLik(object))
  2 * (saturated.llk - fitted.llk)
}


model.matrix.brlm <- function(object, ...) {
  model.matrix( object$terms, object$model )
}


loocv.brlm <- function(object, ...) {
  y <- model.response(object$model)
  x <- model.matrix(object$terms, object$model)
  n <- length(y)
  sigma.sq <- sigma(object)^2
  scl <- (object$nu + 1) / (object$nu * sigma.sq)
  r <- c( y - x %*% coef(object) )
  r.ij <- numeric(n)
  for (i in 1:n) {
    dw <- -object$weights[i]
    db <- scl * c( x[i,] * (r[i] * dw) ) +
      -(coef(object) - object$mu0) / (sigma.sq * object$tau.sq)
    b.ij <- coef(object) + c( object$vcov %*% db )
    r.ij[i] <- c( y[i] - x[i,, drop = FALSE] %*% b.ij )
  }
  sigma.ij <- sd(r.ij)
  -2 * sum(dt(r.ij/sigma.ij, df = object$nu, log = TRUE))
}


summary.brlm <- function(object, ...) {
  res <- structure(
    list(
      call = object$call,
      terms = object$terms,
      residuals = residuals(object),
      sigma = sigma(object),
      nu = object$nu,
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
  ## Compute R^2, deviance
  p <- length(object$coefficients)  # approximate
  n <- nrow(object$model)
  res$df <- c( p, n - p )
  y <- model.response(object$model)  # Null model
  nu <- object$nu
  fit0 <- brlm(y ~ 1, nu = nu)
  res$r.squared = 1 - sigma(object)^2 / sigma(fit0)^2
  res$deviance <- deviance(object)
  res$null.deviance <- deviance(fit0)
  ##
  res
}


print.summary.brlm <- function(x, digits = 3L, ...) {
  cat("\nCall:\n", paste(deparse(x$call), sep = "\n", collapse = "\n"),
      "\n\n", sep = "")
  cat("Residuals:\n")
  rq <- structure(zapsmall(quantile(x$residuals)),
                  names = c("Min", "1Q", "Median", "3Q", "Max"))
  print(rq, digits = digits)
  cat("\n")
  printCoefmat(x$coefficients, digits = digits)
  rse <- if (x$nu <= 2) Inf else sqrt(x$nu / (x$nu - 2)) * x$sigma
  cat("\nResidual standard error:",
      formatC(rse, digits = digits),
      "on", x$df[2L], "degrees of freedom\n")
  cat("R-squared:", formatC(x$r.squared, digits = digits), "\n")
  cat("\n")
  invisible(x)
}




