
#include <cmath>
#include <limits>
#include <string>

#include <Eigen/Core>
#include <Eigen/Dense>



#ifndef _VB_NEWTON_RAPHSON_
#define _VB_NEWTON_RAPHSON_


namespace vb {


  class optim_base {
  public:
    struct control {
      int maxit = 50;
      double xtol_abs = 1e-5;
      double xtol_rel = 1e-10;
    };

    enum class flag {
      none,
      did_not_converge,
      hessian_singular
    };

    struct result {
      int iter = 0;
      flag flag = flag::none;
    };
  };
  

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
	norms[0] = (double)m.linear_update(m.negative_hessian().llt().solve(g));
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



  std::string message(const vb::optim_base::flag f) {
    std::string m = "";
    switch (f) {
    case vb::optim_base::flag::did_not_converge :
      m = "Algorithm did not converge"; break;
    case vb::optim_base::flag::hessian_singular :
      m = "Singular Hessian matrix"; break;
    default: break;
    };
    return m;
  };

  std::string message(const vb::optim_base::result r) {
    std::string m = message(r.flag);
    switch (r.flag) {
    case vb::optim_base::flag::did_not_converge :
      m += " after " + std::to_string(r.iter) + " iterations";
      break;
    default: break;
    };
    return m;
  };


  

}

#endif  // _VB_NEWTON_RAPHSON_

