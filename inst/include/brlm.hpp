
#include <cassert>
#include <cmath>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Dense>
// #include <Eigen/QR>

#include "regression_data.hpp"


#ifndef _VB_BRLM_
#define _VB_BRLM_

namespace vb {

  template< typename T = double >
  class brlm {
  public:
    typedef T scalar_type;
    using param_type = typename vb::glm_data<T>::vector_type;
    using matrix_type = typename vb::glm_data<T>::matrix_type;

    brlm( vb::glm_data<T>& d ) : data_(d) { ; }

    void initialize();
    // void update_sigma();

    /* Update method for any dispersion or kurtosis parameters */
    void update_aux( const param_type& w );
    void beta( const param_type& b );
    
    /* Unnormalized log posterior */
    scalar_type objective() const;
    scalar_type sigma() const { return std::sqrt(sigmasq_); };
    
    param_type fitted() const;
    param_type residuals() const;
    param_type weights() const;

    matrix_type vcov() const;
    matrix_type information() const;

    
    const param_type& pseudo_data() const;
    const param_type& prior_variance() const;
    const matrix_type& x() const;

    const param_type& beta() const { return beta_; };
    param_type& beta() {
      return const_cast<param_type&>(
        const_cast<const brlm<T>*>(this)->beta()
      );
    };
    

  protected:
    param_type beta_;
    T sigmasq_;

    vb::glm_data<T>& data_;
  };
  
}



template< typename T >
void vb::brlm<T>::initialize() {
  const int scl = (data_.n() > data_.p()) ?
    (data_.n() - data_.p()) : data_.n();
  beta_ = data_.start();
  sigmasq_ = residuals().squaredNorm() / scl;
};


template< typename T >
void vb::brlm<T>::beta(
  const vb::brlm<T>::param_type& b
) {
  assert(b.size() == beta_.size() && "Parameter dimension mismatch");
  beta_ = b;
};


template< typename T >
void vb::brlm<T>::update_aux(
  const vb::brlm<T>::param_type& w
) {
  assert(w.size() == data_.n() && "Weights dimension mismatch");
  /* Need to save the weights beforehand to avoid scenario where beta 
   * and sigma are updated using different weights within the same 
   * iteration: weights() computes on the fly
   */
  sigmasq_ = ( w.array() * residuals().array().square() )
    .sum() / data_.n();
};



template< typename T >
typename vb::brlm<T>::scalar_type
vb::brlm<T>::objective() const {
  const scalar_type nu = data_.nu();
  const scalar_type nusig = nu * sigmasq_;
  const scalar_type llk = -data_.n()/2 * std::log(sigmasq_) +
    -(nu+1)/2 * ( residuals().array().abs2() / nusig ).log1p().sum();
  const scalar_type lp = -0.5 / sigmasq_ * (
    data_.tausq().cwiseSqrt().cwiseInverse().asDiagonal() *
    (beta_ - data_.mu())
  ).squaredNorm();
  return llk + lp;
};


template< typename T >
typename vb::brlm<T>::param_type
vb::brlm<T>::weights() const {
  return (data_.nu() + 1) *
    ( data_.nu() + residuals().array().square() / sigmasq_ )
    .inverse().matrix();
};





template< typename T >
typename vb::brlm<T>::param_type
vb::brlm<T>::fitted() const {
  return data_.x() * beta_;
};


template< typename T >
const typename vb::brlm<T>::param_type&
vb::brlm<T>::pseudo_data() const {
  return data_.y();
};


template< typename T >
const typename vb::brlm<T>::param_type&
vb::brlm<T>::prior_variance() const {
  return data_.tausq();
};


template< typename T >
const typename vb::brlm<T>::matrix_type&
vb::brlm<T>::x() const {
  return data_.x();
};



template< typename T >
typename vb::brlm<T>::param_type
vb::brlm<T>::residuals() const {
  return data_.y() - fitted();
};


template< typename T >
typename vb::brlm<T>::matrix_type
vb::brlm<T>::vcov() const {
  const int p = data_.p();
  // const scalar_type nu = data_.nu();
  // Eigen::HouseholderQR<matrix_type> q( data_.x() );
  // matrix_type rinv = q.matrixQR()
  //   .topLeftCorner(p, p)
  //   .template triangularView<Eigen::Upper>()
  //   .solve( matrix_type::Identity(p, p) );
  // return sigmasq_ * (nu + 3) / (nu + 1) * ( rinv * rinv.transpose() );
  return information().llt().solve( matrix_type::Identity(p, p) );
};


template< typename T >
typename vb::brlm<T>::matrix_type
vb::brlm<T>::information() const {
  const scalar_type nu = data_.nu();
  // return (nu + 1) / (nu + 3) / sigmasq_ *
  //   ( data_.x().transpose() * data_.x() );
  matrix_type i = (nu + 1) / (nu + 3) / sigmasq_ *
    ( data_.x().transpose() * data_.x() );
  i.diagonal() += data_.tausq().cwiseInverse();
  return i;
};



#endif  // _VB_BRLM_







// template< typename T >
// typename vb::brlm<T>::matrix_type
// vb::brlm<T>::negative_hessian() const {
//   const int p = beta_.size();
//   const scalar_type nusig = data_.nu() * sigmasq_;
//   const scalar_type scl = (data_.nu() + 1) / nusig;
//   param_type r2 = residuals().cwiseAbs2();
//   param_type w = ( r2.array() / nusig + 1 ).inverse().matrix();
//   param_type dw = (2 / nusig) * (r2.asDiagonal() * w.cwiseAbs2());
//   matrix_type h( p+1, p+1 );
//   h.topLeftCorner(p, p) = scl * data_.x().adjoint() *
//     (w + dw).asDiagonal() * data_.x();
//   h.diagonal().head(p) += ( sigmasq_ * data_.tausq() ).cwiseInverse();
//   h(p, p) = (data_.nu() + 1) * 2 * nusig *
//     ( r2.array() / (nusiig + r2.array()).square() ).sum();
//   return h;
// };


// template< typename T >
// typename vb::brlm<T>::param_type
// vb::brlm<T>::gradient() const {
//   const int p = beta_.size();
//   const int n = data_.n();
//   const scalar_type nusig = data_.nu() * sigmasq_;
//   const scalar_type scl = (data_.nu() + 1) / nusig;
//   param_type r = residuals();
//   param_type g(p + 1);
//   g.head(p) =
//     scl * ( data_.x().adjoint() * (weights().asDiagonal() * r) ) +
//     ( sigmasq_ * data_.tausq() ).cwiseInverse().asDiagonal() *
//     ( data_.mu() - beta_ );
//   g.tail(1) = -(n-1) + (data_.nu() + 1) *
//     (r.array().square() / (nusig + r.array().square())).sum();
//   return g;
// };
