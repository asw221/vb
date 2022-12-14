
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
    vb::newton opt;
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
    // Rcpp::Rcout << "start:\n" << data.start() << "\n\n"
    // 		<< "gradient:\n" << model.gradient() << "\n\n"
    // 		<< "-hessian:\n" << model.negative_hessian() << "\n\n"
    // 		<< std::endl;
    // vb::brlm<>::param_type delta = model.negative_hessian().ldlt().solve(model.gradient());
    // Rcpp::Rcout << "delta:\n"
    // 		<< delta
    // 		<< "\n" << std::endl;
    //
    vb::optim_base::result res = opt.optimize( model, ctrl );
    //
    return Rcpp::wrap(
      Rcpp::List::create(
        Rcpp::Named("coefficients") = model.beta(),
	Rcpp::Named("vcov") = model.negative_hessian().inverse(),
	Rcpp::Named("sigma") = model.sigma(),
	Rcpp::Named("weights") = model.weights(),
	Rcpp::Named("deriv") = model.gradient(),
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
