
#include <cassert>
#include <cmath>

#include <Eigen/Core>

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
    void update_sigma();
    
    scalar_type linear_update( const param_type& delta );
    
    /* Unnormalized log posterior */
    scalar_type objective() const;
    scalar_type sigma() const { return std::sqrt(sigmasq_); };
    
    param_type fitted() const;
    param_type gradient() const;
    param_type residuals() const;
    param_type weights() const;

    matrix_type negative_hessian() const;

    const param_type& beta() const { return beta_; };
    

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
typename vb::brlm<T>::scalar_type
vb::brlm<T>::linear_update(
  const vb::brlm<T>::param_type& delta
) {
  assert( delta.size() == beta_.size() &&
	  "Gradient dimension mismatch" );
  beta_ += delta;
  update_sigma();
  return delta.norm();
};



template< typename T >
typename vb::brlm<T>::scalar_type
vb::brlm<T>::objective() const {
  const scalar_type nu = data_.nu();
  const scalar_type nusig = nu * sigmasq_;
  const scalar_type llk = -data_.n()/2 * std::log(sigmasq_) +
    -(nu+1)/2 * ( residuals().array().abs2() / nusig ).log1p().sum();
  const scalar_type lp = -0.5 * (
    data_.tausq().cwiseSqrt().cwiseInverse().asDiagonal() *
    (beta_ - data_.mu())
  ).squaredNorm();
  return llk + lp;
};


template< typename T >
typename vb::brlm<T>::param_type
vb::brlm<T>::weights() const {
  const scalar_type nusig = data_.nu() * sigmasq_;
  param_type r2 = residuals().cwiseAbs2();
  return ( r2.array() / nusig + 1 ).inverse().matrix();
};



template< typename T >
typename vb::brlm<T>::matrix_type
vb::brlm<T>::negative_hessian() const {
  const scalar_type nusig = data_.nu() * sigmasq_;
  const scalar_type scl = (data_.nu() + 1) / nusig;
  param_type r2 = residuals().cwiseAbs2();
  param_type w = ( r2.array() / nusig + 1 ).inverse().matrix();
  param_type dw = (2 / nusig) * (r2.asDiagonal() * w.cwiseAbs2());
  matrix_type h = scl * data_.x().adjoint() *
    (w + dw).asDiagonal() * data_.x();
  h.diagonal() += ( sigmasq_ * data_.tausq() ).cwiseInverse();
  return h;
};


template< typename T >
typename vb::brlm<T>::param_type
vb::brlm<T>::gradient() const {
  const scalar_type nusig = data_.nu() * sigmasq_;
  const scalar_type scl = (data_.nu() + 1) / nusig;
  param_type r = residuals();
  param_type g =
    scl * ( data_.x().adjoint() * (weights().asDiagonal() * r) ) +
    ( sigmasq_ * data_.tausq() ).cwiseInverse().asDiagonal() *
    ( data_.mu() - beta_ );
  return g;
};




template< typename T >
void vb::brlm<T>::update_sigma() {
  const int scl = (data_.n() > data_.p()) ?
    (data_.n() - data_.p()) : data_.n();
  sigmasq_ = ( weights().cwiseSqrt().asDiagonal() * residuals() )
    .squaredNorm() / scl;
};


template< typename T >
typename vb::brlm<T>::param_type
vb::brlm<T>::fitted() const {
  return data_.x() * beta_;
};


template< typename T >
typename vb::brlm<T>::param_type
vb::brlm<T>::residuals() const {
  return data_.y() - fitted();
};

#endif  // _VB_BRLM_

