
#include <cmath>
#include <limits>
#include <string>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "optim_base.hpp"



#ifndef _VB_NEWTON_RAPHSON_
#define _VB_NEWTON_RAPHSON_


namespace vb {
  

  class newton :
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
       * norms[1] = norms[0] / || grad_theta( f(theta_t) ) ||
       *
       * target[0] = f(theta_{t-1})
       * target[1] = f(theta_t)
       */
      target[0] = -std::numeric_limits<double>::infinity();
      do {
	param_type g = m.gradient();
	param_type delta = m.information().llt().solve(g);
	m.beta( m.beta() + delta );
	m.update_aux();
	// norms[0] = (double)m.linear_update(m.negative_hessian().llt().solve(g));
	norms[0] = delta.norm();
	target[1] = (double)m.objective();
	//
	norms[1] = norms[0] / std::sqrt(g.squaredNorm() + eps);
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

