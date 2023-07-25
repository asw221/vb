
#include <cassert>
#include <cmath>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Dense>
// #include <Eigen/QR>

#include "regression_data.hpp"


#ifndef _VB_BLR_
#define _VB_BLR_

namespace vb {

  // --- blr ---------------------------------------------------------
  ////////////////////////////////////////////////////////////////////
  template< typename T = double >
  class blr {
  public:
    typedef T scalar_type;
    using param_type = typename vb::glm_data<T>::vector_type;
    using matrix_type = typename vb::glm_data<T>::matrix_type;

    blr( vb::glm_data<T>& d ) : data_(d) { ; }

    void initialize();

    /* Update methods */
    void beta( const param_type& b );
    /* Update method for any dispersion or kurtosis parameters */
    void update_aux() { ; }
    //
    
    /* Unnormalized log posterior */
    scalar_type objective() const;
    scalar_type dispersion() const { return psi_; };
    
    param_type fitted() const;
    param_type residuals() const;
    param_type weights() const;

    param_type linear_predictor() const;
    param_type gradient() const;

    matrix_type vcov() const;
    matrix_type information() const;

    const matrix_type& x() const;

    const param_type& beta() const { return beta_; };
    param_type& beta() {
      return const_cast<param_type&>(
        const_cast<const blr<T>*>(this)->beta()
      );
    };
    

  protected:
    param_type beta_;
    T psi_;  // <- Unused dispersion parameter

    vb::glm_data<T>& data_;
  };
  ////////////////////////////////////////////////////////////////////
  
}



template< typename T >
void vb::blr<T>::initialize() {
  beta_ = data_.start();
  //
  psi_ = 1;
};


template< typename T >
void vb::blr<T>::beta(
  const vb::blr<T>::param_type& b
) {
  assert(b.size() == beta_.size() && "Parameter dimension mismatch");
  beta_ = b;
};




template< typename T >
typename vb::blr<T>::scalar_type
vb::blr<T>::objective() const {
  const scalar_type nu = data_.nu();
  const int p = data_.p();
  const param_type eta = linear_predictor();
  const scalar_type b_mu_tau_qf =
    ( (beta_ - data_.mu()).asDiagonal() *
      data_.tausq().cwiseSqrt().cwiseInverse() ).squaredNorm();
  const scalar_type llk =
    ( (data_.y().array() - 1) * eta.array() ).sum() -
    (-eta).array().exp().log1p().sum();
  const scalar_type lp = -0.5 * (nu + p) *
    std::log1p( b_mu_tau_qf / nu );
  return llk + lp;
};


template< typename T >
typename vb::blr<T>::param_type
vb::blr<T>::weights() const {
  return param_type::Ones( data_.n() );
};





template< typename T >
typename vb::blr<T>::param_type
vb::blr<T>::linear_predictor() const {
  return data_.x() * beta_;
};



template< typename T >
typename vb::blr<T>::param_type
vb::blr<T>::fitted() const {
  param_type mu = linear_predictor();
  for ( int i = 0; i < mu.size(); i++ ) {
    scalar_type e_eta = std::exp( -mu[i] );
    mu[i] = e_eta / (1 + e_eta);
  }
  return mu;
};



template< typename T >
typename vb::blr<T>::param_type
vb::blr<T>::gradient() const {
  const scalar_type nu = data_.nu();
  const int p = data_.p();
  const param_type b_mu = beta_ - data_.mu();
  const param_type tausq_inv = data_.tausq().cwiseInverse();
  const scalar_type a = 1 +
    (b_mu.cwiseAbs2().transpose() * tausq_inv)[0] / nu;
  // Compute gradient of log-likelihood
  const param_type w = ( data_.y().array() - 1 +
    (1 + linear_predictor().array().exp()).inverse() ).matrix();
  param_type g = data_.x().transpose() * w;
  // Compute and add gradient of log-prior
  g += ( -(1 + p/nu) / a * b_mu ).asDiagonal() * tausq_inv;
  return g;
};





template< typename T >
const typename vb::blr<T>::matrix_type&
vb::blr<T>::x() const {
  return data_.x();
};



template< typename T >
typename vb::blr<T>::param_type
vb::blr<T>::residuals() const {
  return data_.y() - fitted();
};


template< typename T >
typename vb::blr<T>::matrix_type
vb::blr<T>::vcov() const {
  const int p = data_.p();
  return information().llt().solve( matrix_type::Identity(p, p) );
};


template< typename T >
typename vb::blr<T>::matrix_type
vb::blr<T>::information() const {
  const scalar_type nu = data_.nu();
  const int p = data_.p();
  const param_type b_mu = beta_ - data_.mu();
  const param_type tausq_inv = data_.tausq().cwiseInverse();
  const scalar_type a = 1 +
    (b_mu.cwiseAbs2().transpose() * tausq_inv)[0] / nu;
  const scalar_type b = (1 + p/nu) / a;
  const param_type exb1p =
    ( linear_predictor().array().exp() + 1 ).matrix();
  const param_type v = exb1p.cwiseInverse() +
    exb1p.cwiseAbs2().cwiseInverse();
  // 
  const matrix_type i_llk =
    data_.x().transpose() * v.asDiagonal() * data_.x();
  matrix_type i_prior = -2 / (nu * a) * b *
    tausq_inv.asDiagonal() * b_mu * b_mu.transpose() *
    tausq_inv.asDiagonal();
  i_prior.diagonal() += b * tausq_inv;
  //
  return i_llk + i_prior;
};



#endif  // _VB_BLR_







