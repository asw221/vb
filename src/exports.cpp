
#include <Rcpp.h>
#include <RcppEigen.h>

#include "vb.hpp"


extern "C" {


  SEXP fit_brlm_cpp(
    const SEXP x_, const SEXP y_,
    const SEXP start_, const SEXP df_,
    const SEXP prior_variance_, const SEXP prior_mean_,
    const SEXP maxit_, const SEXP tol_,
    const SEXP reltol_
  ) {
    vb::glm_data<> data;
    vb::optim_base::control ctrl;
    vb::irwls opt;
    //
    vb::initialize_data_r_pointers(
      data, x_, y_, start_, prior_mean_, prior_variance_,
      1.0,  // "phi" parameter is unused here
      Rcpp::as<double>(df_)
    );
    //
    ctrl.maxit = Rcpp::as<int>(maxit_);
    ctrl.xtol_abs = Rcpp::as<double>(tol_);
    ctrl.xtol_rel = Rcpp::as<double>(reltol_);
    //
    vb::brlm<> model(data);
    model.initialize();
    //
    vb::optim_base::result res = opt.optimize( model, ctrl );
    //
    return Rcpp::wrap(
      Rcpp::List::create(
        Rcpp::Named("coefficients") = model.beta(),
	Rcpp::Named("vcov") = model.vcov(),
	Rcpp::Named("sigma") = model.sigma(),
	Rcpp::Named("weights") = model.weights(),
	Rcpp::Named("converged") = res.flag == vb::optim_base::flag::none,
	Rcpp::Named("iter") = res.iter,
	Rcpp::Named("tau.sq") = data.tausq(),
	Rcpp::Named("nu") = data.nu(),
	Rcpp::Named("mu0") = data.mu(),
	Rcpp::Named("residuals") = model.residuals(),
	Rcpp::Named("fitted.values") = model.fitted()
    ));
  };



  SEXP fit_blr_cpp(
    const SEXP x_, const SEXP y_,
    const SEXP start_, const SEXP df_,
    const SEXP prior_scale_, const SEXP prior_location_,
    const SEXP maxit_, const SEXP tol_,
    const SEXP reltol_
  ) {
    vb::glm_data<> data;
    vb::optim_base::control ctrl;
    vb::newton opt;
    //
    vb::initialize_data_r_pointers(
      data, x_, y_, start_, prior_location_, prior_scale_,
      1.0,  // "phi" parameter is unused here
      Rcpp::as<double>(df_)
    );
    //
    ctrl.maxit = Rcpp::as<int>(maxit_);
    ctrl.xtol_abs = Rcpp::as<double>(tol_);
    ctrl.xtol_rel = Rcpp::as<double>(reltol_);
    //
    vb::blr<> model(data);
    model.initialize();
    //
    vb::optim_base::result res = opt.optimize( model, ctrl );
    //
    return Rcpp::wrap(
      Rcpp::List::create(
        Rcpp::Named("coefficients") = model.beta(),
	Rcpp::Named("vcov") = model.vcov(),
	Rcpp::Named("converged") = res.flag == vb::optim_base::flag::none,
	Rcpp::Named("iter") = res.iter,
	Rcpp::Named("tau.sq") = data.tausq(),
	Rcpp::Named("nu") = data.nu(),
	Rcpp::Named("mu0") = data.mu(),
	Rcpp::Named("residuals") = model.residuals(),
	Rcpp::Named("fitted.values") = model.fitted()
    ));
  };

  
}  // extern "C"
