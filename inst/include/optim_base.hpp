
#include <string>


#ifndef _VB_OPTIM_BASE_
#define _VB_OPTIM_BASE_

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


}  // namespace vb


#endif  // _VB_OPTIM_BASE_
