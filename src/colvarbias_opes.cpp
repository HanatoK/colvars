#include "colvarbias_opes.h"
#include "colvarbias.h"
#include "colvarproxy.h"

colvarbias_opes::colvarbias_opes(char const *key):
  colvarbias(key), m_barrier(0), m_biasfactor(0),
  m_bias_prefactor(0), m_temperature(0),
  m_pace(0), m_adaptive_sigma_stride(0),
  m_adaptive_counter(0), m_counter(1), m_compression_threshold(0),
  m_adaptive_sigma(false), m_epsilon(0), m_sum_weights(0),
  m_sum_weights2(0), m_cutoff(0), m_cutoff2(0),
  m_zed(1), m_kdenorm(0), m_val_at_cutoff(0)
{
  enable(f_cvb_scalar_variables);
}

int colvarbias_opes::init(const std::string& conf) {
  int error_code = colvarbias::init(conf);
  m_temperature = cvm::proxy->target_temperature();
  get_keyval(conf, "barrier", m_barrier);
  if (m_barrier < 0) {
    return cvm::error("the barrier should be greater than zero", COLVARS_INPUT_ERROR);
  }
  std::string biasfactor_str;
  get_keyval(conf, "biasfactor", biasfactor_str);
  const cvm::real kbt = m_temperature * cvm::proxy->boltzmann();
  m_biasfactor = m_barrier / kbt;
  if (biasfactor_str == "inf" || biasfactor_str == "INF") {
    m_biasfactor = std::numeric_limits<double>::infinity();
    m_bias_prefactor = -1;
  } else {
    if (biasfactor_str.size() > 0) {
      try {
        m_biasfactor = std::stod(biasfactor_str);
      } catch (const std::exception& e) {
        return cvm::error(e.what(), COLVARS_INPUT_ERROR);
      }
    }
    if (m_biasfactor <= 1.0) {
      return cvm::error("biasfactor must be greater than one (use \"inf\" for uniform target)");
    }
    m_bias_prefactor = 1 - 1.0 / m_biasfactor;
  }
  get_keyval(conf, "adaptive_sigma", m_adaptive_sigma, false);
  if (m_adaptive_sigma) {
    get_keyval(conf, "adaptive_sigma_stride", m_adaptive_sigma_stride, 0);
    if (std::isinf(m_biasfactor)) {
      return cvm::error("cannot use infinite biasfactor with adaptive sigma",
                        COLVARS_INPUT_ERROR);
    }
    if (m_adaptive_sigma_stride == 0) {
      m_adaptive_sigma_stride = time_step_factor * 10;
    }
  } else {
    get_keyval(conf, "sigma", m_sigma0);
    if (m_sigma0.size() != num_variables()) {
      return cvm::error("number of sigma parameters does not match the number of variables",
                        COLVARS_INPUT_ERROR);
    }
    get_keyval(conf, "sigma_min", m_sigma_min);
    if ((m_sigma_min.size() != 0) && (m_sigma_min.size() != num_variables())) {
      return cvm::error("incorrect number of parameters of sigma_min");
    }
    for (size_t i = 0; i < num_variables(); ++i) {
      if (m_sigma_min[i] > m_sigma0[i]) {
        return cvm::error("sigma_min of variable " + cvm::to_str(i) + " should be smaller than sigma");
      }
    }
  }
  get_keyval(conf, "epsilon", m_epsilon, std::exp(-m_barrier/m_bias_prefactor/kbt));
  if (m_epsilon <= 0) {
    return cvm::error("you must choose a value of epsilon greater than zero");
  }
  m_sum_weights = std::pow(m_epsilon, m_bias_prefactor);
  m_sum_weights2 = m_sum_weights * m_sum_weights;
  get_keyval(conf, "kernel_cutoff", m_cutoff, std::sqrt(2.0*m_barrier/m_bias_prefactor/kbt));
  if (m_cutoff <= 0) {
    return cvm::error("you must choose a value of kernel_cutoff greater than zero");
  }
  m_cutoff2 = m_cutoff * m_cutoff;
  m_val_at_cutoff = std::exp(-0.5 * m_cutoff2);
  get_keyval(conf, "compression_threshold", m_compression_threshold, 1);
  if (m_compression_threshold != 0) {
    if (m_compression_threshold > m_cutoff) {
      return cvm::error("compression_threshold cannot be larger than kernel_cutoff", COLVARS_INPUT_ERROR);
    }
  }
  get_keyval(conf, "pace", m_pace);
  m_av_cv.assign(num_variables(), 0);
  m_av_M2.assign(num_variables(), 0);
  return error_code;
}

cvm::real colvarbias_opes::evaluateKernel(
  const colvarbias_opes::kernel& G,
  const std::vector<cvm::real>& x) const {
  cvm::real norm2 = 0;
  for (size_t i = 0; i < num_variables(); ++i) {
    const cvm::real dist2_i = variables(i)->dist2(G.m_center[i], x[i]) / (G.m_sigma[i] * G.m_sigma[i]);
    norm2 += dist2_i;
    if (norm2 >= m_cutoff2) {
      return 0;
    }
  }
  return G.m_height * (std::exp(-0.5 * norm2) - m_val_at_cutoff);
}

cvm::real colvarbias_opes::evaluateKernel(
  const colvarbias_opes::kernel& G,
  const std::vector<cvm::real>& x,
  std::vector<cvm::real>& accumulated_derivative,
  std::vector<cvm::real>& dist) const {
  cvm::real norm2 = 0;
  for (size_t i = 0; i < num_variables(); ++i) {
    // const cvm::real dist2_i = variables(i)->dist2(G.m_center[i], x[i]) / (G.m_sigma[i] * G.m_sigma[i]);
    dist[i] = variables(i)->dist2_lgrad(x[i], G.m_center[i]) / G.m_sigma[i];
    norm2 += dist[i] * dist[i];
    if (norm2 >= m_cutoff2) {
      return 0;
    }
  }
  const cvm::real val = G.m_height * (std::exp(-0.5 * norm2) - m_val_at_cutoff);
  // The derivative of norm2 with respect to x
  for (size_t i = 0; i < num_variables(); ++i) {
    accumulated_derivative[i] -= val * dist[i] / G.m_sigma[i];
  }
  return val;
}

cvm::real colvarbias_opes::getProbAndDerivatives(
  const std::vector<cvm::real>& cv, std::vector<cvm::real>& der_prob,
  std::vector<cvm::real>& dist) {
  double prob = 0.0;
  // TODO: implement neighbor list to accelerate the calculation
  // TODO: PLUMED uses openmp. What should I use to accelerate the loop?
  for (size_t k = 0; k < m_kernels.size(); ++k) {
    prob += evaluateKernel(m_kernels[k], cv, der_prob, dist);
  }
  prob /= m_kdenorm;
  for (size_t i = 0; i < num_variables(); ++i) {
    der_prob[i] /= m_kdenorm;
  }
  return prob;
}

int colvarbias_opes::update() {
  int error_code = COLVARS_OK;
  // forward
  std::vector<cvm::real> cv(num_variables());
  for (size_t i = 0; i < num_variables(); ++i) {
    cv[i] = variables(i)->value();
  }
  std::vector<cvm::real> der_prob(num_variables(), 0);
  std::vector<cvm::real> cv_dist(num_variables(), 0);
  const cvm::real prob = getProbAndDerivatives(cv, der_prob, cv_dist);
  const cvm::real kbt = cvm::proxy->target_temperature() * cvm::proxy->boltzmann();
  const cvm::real bias = kbt * m_bias_prefactor * cvm::logn(prob / m_zed + m_epsilon);
  bias_energy = bias;
  for (size_t i = 0; i < num_variables(); ++i) {
    // TODO: check the gradient
    colvar_forces[i] = -kbt * m_bias_prefactor / (prob / m_zed + m_epsilon) * der_prob[i] / m_zed;
  }
  // backward
  if (m_adaptive_sigma) {
    m_adaptive_counter++;
    cvm::step_number tau = m_adaptive_sigma_stride;
    if (m_adaptive_counter < m_adaptive_sigma_stride) tau = m_adaptive_counter;
    for (size_t i = 0; i < num_variables(); ++i) {
      // Welford's online algorithm for standard deviation
      const cvm::real diff_i = 0.5 * variables(i)->dist2_lgrad(cv[i], m_av_cv[i]);
      m_av_cv[i] += diff_i / tau;
      m_av_M2[i] += diff_i * 0.5 * variables(i)->dist2_lgrad(cv[i], m_av_cv[i]);
    }
    if (m_adaptive_counter <m_adaptive_sigma_stride/* && restarting*/) {
      return error_code;
    }
  }
  // TODO: check MTS?
  const cvm::real old_kdenorm = m_kdenorm;
  // TODO: how could I account for extra biases in Colvars?
  cvm::real log_weight = bias / kbt;
  cvm::real height = cvm::exp(log_weight);
  // TODO: sum heights from other replicas.
  m_counter += 1;
  m_sum_weights += height;
  m_sum_weights2 += height * height;
  m_neff = (1 + m_sum_weights) * (1 + m_sum_weights) / (1 + m_sum_weights2);
  m_rct = kbt * cvm::logn(m_sum_weights / m_counter);
  m_kdenorm = m_sum_weights;
  std::vector<cvm::real> sigma = m_sigma0;
  if (m_adaptive_sigma) {
    const cvm::real factor = m_biasfactor;
    // TODO
  }
  return error_code;
}
