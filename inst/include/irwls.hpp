
#include <cmath>
#include <limits>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/QR>


#include "optim_base.hpp"



#ifndef _VB_IRWLS_
#define _VB_NEWTON_RAPHSON_


namespace vb {

  Eigen::VectorXd scalar_wls_solve(
    const Eigen::MatrixXd& x,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& w
  ) {
    Eigen::VectorXd w_sqrt = w.cwiseAbs().cwiseSqrt();
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> q(
      w_sqrt.asDiagonal() * x
    );
    return q.solve( w_sqrt.asDiagonal() * y );
  };

  /* WLS, scalar outcome, diagonal prior variance */
  Eigen::VectorXd scalar_wls_solve_dvprior(
    const Eigen::MatrixXd& x,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& w,
    const Eigen::VectorXd& v  /*< Prior variance */
  ) {
    Eigen::MatrixXd h = x.transpose() * w.asDiagonal() * x;
    h.diagonal() += v.cwiseInverse();
    Eigen::LLT<Eigen::MatrixXd> decomp( h );
    return decomp.solve( x.transpose() * (w.asDiagonal() * y) );
  };


  
  class irwls :
    public optim_base {
  public:
    template< typename Model >
    optim_base::result
    optimize(
      Model& m,
      const optim_base::control control
    ) const {
      using param_type = typename Model::param_type;
      vb::optim_base::result r;
      const double eps = std::numeric_limits<double>::epsilon();
      bool converged = false;
      double norms[2], target[2];
      /* 
       * norms[0] = || theta_t - theta_{t-1} ||
       * norms[1] = norms[0] / || theta_{t-1} ||
       *
       * target[0] = f(theta_{t-1})
       * target[1] = f(theta_t)
       */
      target[0] = -std::numeric_limits<double>::infinity();
      do {
	param_type beta0 = m.beta();
	param_type w = m.weights();
	m.beta( vb::scalar_wls_solve_dvprior(
          m.x(), m.pseudo_data(), w, m.prior_variance()
        ) );
	m.update_aux( w );
	norms[0] = (double)( m.beta() - beta0 ).norm();
	target[1] = (double)m.objective();
	//
	norms[1] = norms[0] / std::sqrt(beta0.squaredNorm() + eps);
	converged = ( norms[0] <= control.xtol_abs ||
		      norms[1] <= control.xtol_rel ) &&
	  ( target[1] >= target[0] );
	//
	target[0] = target[1];
      }
      while ( ++r.iter < control.maxit && !converged );
      //
      if ( !converged )
	r.flag = optim_base::flag::did_not_converge;
      return r;
    }
  };


  

}

#endif  // _VB_NEWTON_RAPHSON_

