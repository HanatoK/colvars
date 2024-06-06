#include "colvarcomp_pucker.h"
#include "colvarmodule.h"
#include <cmath>

struct cpABC {
  cvm::real A;
  cvm::real B;
  cvm::real C;
};

struct cpABC_grad {
  std::vector<cvm::rvector>& dA_dr;
  std::vector<cvm::rvector>& dB_dr;
  std::vector<cvm::rvector>& dC_dr;
};

cpABC calc_cpABC(const cvm::atom_group& r, cpABC_grad* grad = nullptr) {
  // Calculate the center of geometry
  cvm::rvector cog{0, 0, 0};
  for (auto it = r.begin(); it != r.end(); ++it) cog += (it->pos);
  cog /= r.size();
  // Calculate R
  std::vector<cvm::rvector> R;
  for (auto it = r.begin(); it != r.end(); ++it) R.push_back(it->pos - cog);
  // Calculate R' and R''
  cvm::rvector Rp{0, 0, 0};
  cvm::rvector Rpp{0, 0, 0};
  // Save the sine and cosine factors for derivative calculations
  std::vector<cvm::real> sin_f1, cos_f1;
  for (size_t i = 0; i < r.size(); ++i) {
    const cvm::real factor = 2.0 * M_PI * i / r.size();
    const cvm::real sin_f = std::sin(factor);
    const cvm::real cos_f = std::cos(factor);
    Rp += sin_f * R[i];
    Rpp += cos_f * R[i];
    if (grad) {
      sin_f1.push_back(sin_f);
      cos_f1.push_back(cos_f);
    }
  }
  // Calculate the normal vector
  const cvm::rvector cross = cvm::rvector::outer(Rp, Rpp);
  const cvm::real denominator = cross.norm();
  const cvm::rvector n_hat = cross / denominator;
  // Project R_j onto the normal vector
  std::vector<cvm::real> z;
  for (size_t i = 0; i < r.size(); ++i) {
    z.push_back(R[i] * n_hat);
  }
  // Calculate A, B and C from z_j
  cpABC result{0, 0, 0};
  // Save the sine and cosine factors for derivative calculations
  std::vector<cvm::real> sin_f2, cos_f2;
  for (size_t i = 0; i < r.size(); ++i) {
    const cvm::real factor = 2.0 * M_PI * 2.0 * i / r.size();
    const cvm::real sin_f = std::sin(factor);
    const cvm::real cos_f = std::cos(factor);
    result.A += z[i] * sin_f;
    result.B += z[i] * cos_f;
    result.C += z[i] * ((-2.0) * (i % 2) + 1.0);
    if (grad) {
      sin_f2.push_back(sin_f);
      cos_f2.push_back(cos_f);
    }
  }
  // Gradients of ABC with respect to r
  if (grad) {
    grad->dA_dr.assign(r.size(), cvm::rvector{0, 0, 0});
    grad->dB_dr.assign(r.size(), cvm::rvector{0, 0, 0});
    grad->dC_dr.assign(r.size(), cvm::rvector{0, 0, 0});
    const cvm::real tmp2 = 1.0 - 1.0 / r.size();
    const cvm::real tmp3 = -1.0 / r.size();
    const cvm::real one_denom = 1.0 / denominator;
    const cvm::real one_denom_sq = one_denom * one_denom;
    std::vector<cvm::real> triples;
    for (size_t k = 0; k < r.size(); ++k) {
      triples.push_back(R[k] * cross);
    }
    for (size_t j = 0; j < r.size(); ++j) {
      // ∂R'/∂rj
      cvm::real dRp_drj_f = 0;
      cvm::real dRpp_drj_f = 0;
      for (size_t k = 0; k < r.size(); ++k) {
        if (j == k) {
          dRp_drj_f  += sin_f1[k] * tmp2;
          dRpp_drj_f += cos_f1[k] * tmp2;
        } else {
          dRp_drj_f  += sin_f1[k] * tmp3;
          dRpp_drj_f += cos_f1[k] * tmp3;
        }
      }
      // ∂R'/∂rj × R''
      const cvm::rvector dRp_drjx_times_Rpp = cvm::rvector::outer({dRp_drj_f, 0, 0}, Rpp);
      const cvm::rvector dRp_drjy_times_Rpp = cvm::rvector::outer({0, dRp_drj_f, 0}, Rpp);
      const cvm::rvector dRp_drjz_times_Rpp = cvm::rvector::outer({0, 0, dRp_drj_f}, Rpp);
      // R' × ∂R''/∂rj
      const cvm::rvector Rp_times_dRpp_drjx = cvm::rvector::outer(Rp, {dRpp_drj_f, 0, 0});
      const cvm::rvector Rp_times_dRpp_drjy = cvm::rvector::outer(Rp, {0, dRpp_drj_f, 0});
      const cvm::rvector Rp_times_dRpp_drjz = cvm::rvector::outer(Rp, {0, 0, dRpp_drj_f});
      // Derivative of the norm
      const cvm::real dnorm_dx = one_denom * ((cross * dRp_drjx_times_Rpp) + (Rp_times_dRpp_drjx * cross));
      const cvm::real dnorm_dy = one_denom * ((cross * dRp_drjy_times_Rpp) + (Rp_times_dRpp_drjy * cross));
      const cvm::real dnorm_dz = one_denom * ((cross * dRp_drjz_times_Rpp) + (Rp_times_dRpp_drjz * cross));
      // ABC derivatives with respect to r
      for (size_t k = 0; k < r.size(); ++k) {
        cvm::real dtriple_dx = 0;
        cvm::real dtriple_dy = 0;
        cvm::real dtriple_dz = 0;
        if (j == k) {
          dtriple_dx = tmp2 * cross.x + R[k] * dRp_drjx_times_Rpp + R[k] * Rp_times_dRpp_drjx;
          dtriple_dy = tmp2 * cross.y + R[k] * dRp_drjy_times_Rpp + R[k] * Rp_times_dRpp_drjy;
          dtriple_dz = tmp2 * cross.z + R[k] * dRp_drjz_times_Rpp + R[k] * Rp_times_dRpp_drjz;
        } else {
          dtriple_dx = tmp3 * cross.x + R[k] * dRp_drjx_times_Rpp + R[k] * Rp_times_dRpp_drjx;
          dtriple_dy = tmp3 * cross.y + R[k] * dRp_drjy_times_Rpp + R[k] * Rp_times_dRpp_drjy;
          dtriple_dz = tmp3 * cross.z + R[k] * dRp_drjz_times_Rpp + R[k] * Rp_times_dRpp_drjz;
        }
        // Derivative of z_j wrt r_j
        const cvm::rvector dzk_drj{one_denom_sq * (dtriple_dx * denominator - dnorm_dx * triples[k]),
                                   one_denom_sq * (dtriple_dy * denominator - dnorm_dy * triples[k]),
                                   one_denom_sq * (dtriple_dz * denominator - dnorm_dz * triples[k])};
        grad->dA_dr[j] += sin_f2[k] * dzk_drj;
        grad->dB_dr[j] += cos_f2[k] * dzk_drj;
        grad->dC_dr[j] += ((-2.0) * (k % 2) + 1.0) * dzk_drj;
      }
    }
  }
  return result;
}

colvar::cpQ::cpQ() {
  set_function_type("cpQ");
  x.type(colvarvalue::type_scalar);
  enable(f_cvc_explicit_gradient);
}

int colvar::cpQ::init(const std::string& conf) {
  int error_code = cvc::init(conf);

  atoms = parse_group(conf, "atoms");
  if (!atoms || atoms->size() != 6) {
    return error_code | COLVARS_INPUT_ERROR;
  }
  return error_code;
}

void colvar::cpQ::calc_value() {
  cpABC_grad grad{dA_dr, dB_dr, dC_dr};
  cpABC result = calc_cpABC(*atoms, &grad);
  A = result.A;
  B = result.B;
  C = result.C;
  x.real_value = std::sqrt((2.0 * A * A + 2.0 * B * B + C * C) / 6.0);
}

void colvar::cpQ::calc_gradients() {
  for (size_t ia = 0; ia < atoms->size(); ia++) {
    (*atoms)[ia].grad = (2.0 * A * dA_dr[ia] + 2.0 * B * dB_dr[ia] + C * dC_dr[ia]) / (6.0 * x.real_value);
  }
}

colvar::cptheta::cptheta() {
  set_function_type("cptheta");
  init_as_angle();
  enable(f_cvc_explicit_gradient);
}

int colvar::cptheta::init(const std::string& conf) {
  int error_code = cvc::init(conf);

  atoms = parse_group(conf, "atoms");
  if (!atoms || atoms->size() != 6) {
    return error_code | COLVARS_INPUT_ERROR;
  }
  return error_code;
}

void colvar::cptheta::calc_value() {
  cpABC_grad grad{dA_dr, dB_dr, dC_dr};
  cpABC result = calc_cpABC(*atoms, &grad);
  A = result.A;
  B = result.B;
  C = result.C;
  x.real_value = 180.0 / M_PI * std::acos(C / std::sqrt(2.0 * A * A + 2.0 * B * B + C * C));
}

void colvar::cptheta::calc_gradients() {
  const cvm::real tmp1 = 2.0 * (A * A + B * B) + C * C;
  const cvm::real factor = -180.0 / M_PI / std::sqrt(1.0 - (C * C / tmp1)) * (1.0 / tmp1);
  const cvm::real tmp2 = std::sqrt(tmp1);
  const cvm::real tmp3 = 1.0 / tmp2;
  for (size_t ia = 0; ia < atoms->size(); ia++) {
    (*atoms)[ia].grad = factor * (dC_dr[ia] * tmp2 - C * tmp3 * (2.0 * A * dA_dr[ia] + 2.0 * B * dB_dr[ia] + C * dC_dr[ia]));
  }
}

colvar::cpphi::cpphi() {
  set_function_type("cpphi");
  x.type(colvarvalue::type_scalar);
  provide(f_cvc_periodic);
  enable(f_cvc_periodic);
  period = 360.0;
  init_scalar_boundaries(0, 360.0);
  enable(f_cvc_explicit_gradient);
}

int colvar::cpphi::init(const std::string& conf) {
  int error_code = cvc::init(conf);

  atoms = parse_group(conf, "atoms");
  if (!atoms || atoms->size() != 6) {
    return error_code | COLVARS_INPUT_ERROR;
  }
  return error_code;
}

void colvar::cpphi::calc_value() {
  cpABC_grad grad{dA_dr, dB_dr, dC_dr};
  cpABC result = calc_cpABC(*atoms, &grad);
  A = result.A;
  B = result.B;
  C = result.C;
  x.real_value = 180.0 / M_PI * std::atan2(-A, B);
  if (x.real_value < 0) {
    x.real_value += 360.0;
  }
}

void colvar::cpphi::calc_gradients() {
  const cvm::real factor = 180.0 / M_PI / (A * A + B * B);
  for (size_t ia = 0; ia < atoms->size(); ia++) {
    (*atoms)[ia].grad = factor * (-dA_dr[ia] * B + dB_dr[ia] * A);
  }
}
