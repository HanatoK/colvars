#include "colvarbias.h"

#include <vector>

// OPES_METAD implementation: swiped from OPESmetad.cpp from PLUMED
// TODO: implement the explore mode
class colvarbias_opes: public colvarbias {
public:
  struct kernel {
    cvm::real m_height;
    std::vector<cvm::real> m_center;
    std::vector<cvm::real> m_sigma;
    kernel(cvm::real h, const std::vector<cvm::real>& c,
           const std::vector<cvm::real>& s):
      m_height(h), m_center(c), m_sigma(s) {}
  };
  colvarbias_opes(char const *key);
  virtual int init(std::string const &conf) override;
  virtual int update() override;
  virtual int calc_energy(std::vector<colvarvalue> const *values) override;
  virtual int calc_forces(std::vector<colvarvalue> const *values) override;
  std::ostream &write_state_data(std::ostream &os) override;
  std::istream &read_state_data(std::istream &is) override;
  cvm::real getProbAndDerivatives(const std::vector<cvm::real>& cv, std::vector<cvm::real>& der_prob, std::vector<cvm::real>& dist);
  cvm::real evaluateKernel(const kernel& G, const std::vector<cvm::real>& x) const;
  cvm::real evaluateKernel(const kernel& G, const std::vector<cvm::real>& x, std::vector<cvm::real>& accumulated_derivative, std::vector<cvm::real>& dist) const;
private:
  cvm::real m_barrier;
  cvm::real m_biasfactor;
  cvm::real m_bias_prefactor;
  cvm::real m_temperature;
  cvm::step_number m_pace;
  cvm::step_number m_adaptive_sigma_stride;
  cvm::step_number m_adaptive_counter;
  cvm::step_number m_counter;
  cvm::real m_compression_threshold;
  bool m_adaptive_sigma;
  std::vector<cvm::real> m_sigma0;
  std::vector<cvm::real> m_sigma_min;
  cvm::real m_epsilon;
  cvm::real m_sum_weights;
  cvm::real m_sum_weights2;
  cvm::real m_cutoff;
  cvm::real m_cutoff2;
  cvm::real m_zed;
  cvm::real m_kdenorm;
  cvm::real m_val_at_cutoff;
  cvm::real m_rct;
  cvm::real m_neff;
  std::vector<kernel> m_kernels;
  std::vector<cvm::real> m_av_cv;
  std::vector<cvm::real> m_av_M2;
  std::ostringstream m_hills_traj;
};
