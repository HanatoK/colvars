#ifndef COLVARCOMP_EIGEN_RMSD
#define COLVARCOMP_EIGEN_RMSD

#include "colvarcomp.h"
#include <memory>

class colvar::eigen_rmsd: public colvar::cvc {
private:
  struct eigen_rmsd_impl_;
  std::unique_ptr<eigen_rmsd_impl_> p_impl;
public:
  eigen_rmsd(const std::string& conf);
  eigen_rmsd();
  virtual ~eigen_rmsd() = default;
  virtual void calc_value();
  virtual void calc_gradients();
  virtual void apply_force(colvarvalue const &cvforce);
};

#endif // COLVARCOMP_EIGEN_RMSD
