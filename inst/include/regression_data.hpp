
#include <string>
#include <vector>

#include <Rcpp.h>
#include <RcppEigen.h>
#include <Eigen/Core>


#ifndef _VB_GLM_DATA_
#define _VB_GLM_DATA_


namespace vb {

  
  template< typename T = double >
  class glm_data {
  public:
    typedef T scalar_type;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, 1> vector_type;
    typedef typename Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>
      matrix_type;

    enum class flag {
      none,
      dimension_mismatch_xy,
      dimension_mismatch_mu,
      dimension_mismatch_tausq,
      dimension_mismatch_start,
      domain_tausq,
      domain_phi,
      domain_nu
    };

    void validate(
      std::vector<typename vb::glm_data<T>::flag>& v
    ) const;
    bool validate() const;

    /* Getters */
    const matrix_type& x() const { return x_; };
    
    const vector_type& mu() const { return mu_; };
    const vector_type& start() const { return start_; };
    const vector_type& tausq() const { return tau_; };
    const vector_type& y() const { return y_; };

    T phi() const { return phi_; };
    T nu() const { return nu_; };
    
    int n() const { return y_.size(); };
    int p() const { return x_.cols(); };

    /* Setters */
    void x( const matrix_type& a ) { x_ = a; };
    
    void mu( const vector_type& a ) { mu_ = a; };
    void start( const vector_type& a ) { start_ = a; };
    void tausq( const vector_type& a ) { tau_ = a; };
    void y( const vector_type& a ) { y_ = a; };

    void phi( const T a ) { phi_ = a; };
    void nu( const T a ) { nu_ = a; };
    
  protected:
    matrix_type x_;
    vector_type y_;

    /* Costant parameters treated like data */
    vector_type tau_;     /* Prior scales^2 */
    vector_type mu_;      /* Prior locations */
    vector_type start_;   /* Starting values */
    T phi_;  /* Likelihood dispersion parameter */
    T nu_;   /* Likelihood auxiliary parameter */
  };



  bool initialize_data_r_pointers(
    vb::glm_data<double>& data,				  
    const SEXP x, const SEXP y,
    const SEXP start, 
    const SEXP prior_mean, const SEXP prior_variance,
    const double phi, const double nu
  );


  template< typename T >
  std::string message(const typename vb::glm_data<T>::flag f );
  
}  // namespace vb






bool vb::initialize_data_r_pointers(
  vb::glm_data<double>& data,
  const SEXP x, const SEXP y,
  const SEXP start, 
  const SEXP prior_mean, const SEXP prior_variance,
  const double phi, const double nu
) {
  using flag_t = typename vb::glm_data<>::flag;
  data.x( Rcpp::as< Eigen::Map<Eigen::MatrixXd> >(x) );
  data.y( Rcpp::as< Eigen::Map<Eigen::VectorXd> >(y) );
  data.start( Rcpp::as< Eigen::Map<Eigen::VectorXd> >(start) );
  data.mu( Rcpp::as< Eigen::Map<Eigen::VectorXd> >(prior_mean) );
  data.tausq( Rcpp::as< Eigen::Map<Eigen::VectorXd> >(prior_variance) );
  data.phi( phi );
  data.nu( nu );
  //
  const bool success = data.validate();
  if ( !success ) {
    std::vector<flag_t> flags;
    data.validate(flags);
    for ( std::vector<flag_t>::iterator fl = flags.begin();
	  fl != flags.end(); ++fl
	  ) {
      Rcpp::Rcout << vb::message<double>( *fl ) << "\n";
    }
  }
  //
  return success;
};




template< typename T >
void vb::glm_data<T>::validate(
  std::vector<typename vb::glm_data<T>::flag>& v
) const {
  v.clear();
  //
  /* Check for any dimension mismatches */
  if ( x_.rows() != y_.size() )
    v.push_back( flag::dimension_mismatch_xy );
  //
  if ( x_.cols() != mu_.size() )
    v.push_back( flag::dimension_mismatch_mu );
  if ( x_.cols() != tau_.size() )
    v.push_back( flag::dimension_mismatch_tausq );
  if ( x_.cols() != start_.size() )
    v.push_back( flag::dimension_mismatch_start );
  //
  /* Check for any domain mismatches */
  for ( int i = 0; i < tau_.size(); i++ ) {
    if ( tau_[i] <= 0 ) {
      v.push_back( flag::domain_tausq );
      break;
    }
  } 
  if ( phi_ <= 0 )
    v.push_back( flag::domain_phi );
  if ( nu_ <= 0 )
    v.push_back( flag::domain_nu );
  //
  if ( v.empty() )
    v.push_back( flag::none );
};


template< typename T >
bool vb::glm_data<T>::validate() const {
  std::vector<flag> v;
  validate(v);
  return v.size() == 1 && v[0] == flag::none;
};




template< typename T >
std::string vb::message(const typename vb::glm_data<T>::flag f ) {
  std::string m = "";
  switch (f) {
  case vb::glm_data<T>::flag::dimension_mismatch_xy :
    m = "Dimension mismatch: x/y";
    break;
  case vb::glm_data<T>::flag::dimension_mismatch_mu :
    m = "Dimension mismatch: prior mean";
    break;
  case vb::glm_data<T>::flag::dimension_mismatch_tausq :
    m = "Dimension mismatch: prior variance";
    break;
  case vb::glm_data<T>::flag::dimension_mismatch_start :
    m = "Dimension mismatch: starting values";
    break;
  case vb::glm_data<T>::flag::domain_tausq :
    m = "Prior variance <= 0";
    break;
  case vb::glm_data<T>::flag::domain_phi :
    m = "Auxiliary parameter phi <= 0";
    break;
  case vb::glm_data<T>::flag::domain_nu :
    m = "Auxiliary parameter nu <= 0";
    break;
  default: break;
  };
  return m;
};


#endif  // _VB_GLM_DATA_
