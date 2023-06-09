#include "colvarcomp_eigen_rmsd.h"
#include <cmath>

// workaround NAMD naming conflicts
#ifdef C1
#undef C1
#endif

#ifdef C2
#undef C2
#endif

#include <Eigen/Dense>

struct colvar::eigen_rmsd::eigen_rmsd_impl_ {
  void copy_ref_pos_to_cache(const std::vector<cvm::atom_pos>& ref_pos) {
    cached_ref_pos.resize(3, ref_pos.size());
    cvm::real ref_center[3] = {0};
    for (size_t ia = 0; ia < ref_pos.size(); ++ia) {
      ref_center[0] += ref_pos[ia].x;
      ref_center[1] += ref_pos[ia].y;
      ref_center[2] += ref_pos[ia].z;
    }
    ref_center[0] /= ref_pos.size();
    ref_center[1] /= ref_pos.size();
    ref_center[2] /= ref_pos.size();
    for (size_t ia = 0; ia < ref_pos.size(); ++ia) {
      cached_ref_pos(0, ia) = ref_pos[ia].x - ref_center[0];
      cached_ref_pos(1, ia) = ref_pos[ia].y - ref_center[1];
      cached_ref_pos(2, ia) = ref_pos[ia].z - ref_center[2];
    }
  }
  void update_current_center() {
    current_center[0] = 0;
    current_center[1] = 0;
    current_center[2] = 0;
    for (size_t ia = 0; ia < atoms->size(); ++ia) {
      current_center[0] += (*atoms)[ia].pos.x;
      current_center[1] += (*atoms)[ia].pos.y;
      current_center[2] += (*atoms)[ia].pos.z;
    }
    current_center[0] /= atoms->size();
    current_center[1] /= atoms->size();
    current_center[2] /= atoms->size();
  }
  cvm::atom_group *atoms;
  cvm::real current_center[3];
  Eigen::Matrix<double, 3, Eigen::Dynamic> cached_ref_pos;
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> m_eigensolver;
  Eigen::Matrix3d m_C;
  Eigen::Matrix4d m_S;
  Eigen::Matrix3d m_rotation_matrix;
  cvm::real m_sum_square;
  eigen_rmsd_impl_();
  eigen_rmsd_impl_(const std::string& conf, colvar::eigen_rmsd* p);
  void calc_value(colvar::eigen_rmsd* p);
  void calc_gradients(colvar::eigen_rmsd* p);
};

colvar::eigen_rmsd::eigen_rmsd_impl_::eigen_rmsd_impl_() {}

colvar::eigen_rmsd::eigen_rmsd_impl_::eigen_rmsd_impl_(const std::string& conf, colvar::eigen_rmsd* p) {
  cvm::log("Using the experimentail EigenRMSD");
  p->set_function_type("EigenRMSD");
  p->init_as_distance();
  atoms = p->parse_group(conf, "atoms");
  if (!atoms || atoms->size() == 0) {
    cvm::error("Error: \"atoms\" must contain at least 1 atom to compute RMSD.");
    return;
  }
  if (atoms->fitting_group != NULL) {
    cvm::error("Error: This colvar component does not support fitting group.");
    return;
  }
  std::vector<cvm::atom_pos> ref_pos;
  if (p->get_keyval(conf, "refPositions", ref_pos, ref_pos)) {
    cvm::log("Using reference positions from configuration file to calculate the variable.\n");
    if (ref_pos.size() != atoms->size()) {
      cvm::error("Error: the number of reference positions provided ("+
                  cvm::to_str(ref_pos.size())+
                  ") does not match the number of atoms of group \"atoms\" ("+
                  cvm::to_str(atoms->size())+").\n");
      return;
    }
  } else { // Only look for ref pos file if ref positions not already provided
    std::string ref_pos_file;
    if (p->get_keyval(conf, "refPositionsFile", ref_pos_file, std::string(""))) {

      if (ref_pos.size()) {
        cvm::error("Error: cannot specify \"refPositionsFile\" and "
                          "\"refPositions\" at the same time.\n");
        return;
      }

      std::string ref_pos_col;
      double ref_pos_col_value=0.0;

      if (p->get_keyval(conf, "refPositionsCol", ref_pos_col, std::string(""))) {
        // if provided, use PDB column to select coordinates
        bool found = p->get_keyval(conf, "refPositionsColValue", ref_pos_col_value, 0.0);
        if (found && ref_pos_col_value==0.0) {
          cvm::error("Error: refPositionsColValue, "
                     "if provided, must be non-zero.\n");
          return;
        }
      }

      ref_pos.resize(atoms->size());

      cvm::load_coords(ref_pos_file.c_str(), &ref_pos, atoms,
                       ref_pos_col, ref_pos_col_value);
    } else {
      cvm::error("Error: no reference positions for RMSD; use either refPositions of refPositionsFile.");
      return;
    }
  }
  if (ref_pos.size() != atoms->size()) {
    cvm::error("Error: found " + cvm::to_str(ref_pos.size()) +
                    " reference positions for RMSD; expected " + cvm::to_str(atoms->size()));
    return;
  }
  if (atoms->b_user_defined_fit) {
    cvm::error("Error: This colvar component does not support user-defined fitting.");
    return;
  }
  copy_ref_pos_to_cache(ref_pos);
}

colvar::eigen_rmsd::eigen_rmsd(): p_impl(std::make_unique<eigen_rmsd_impl_>()) {}

colvar::eigen_rmsd::eigen_rmsd(const std::string& conf): cvc(conf) {
  p_impl = std::make_unique<eigen_rmsd_impl_>(conf, this);
}

void colvar::eigen_rmsd::eigen_rmsd_impl_::calc_value(colvar::eigen_rmsd* p) {
  // covariance matrix
  m_C = Eigen::Matrix3d::Zero();
  m_sum_square = 0;
  update_current_center();
  for (size_t ia = 0; ia < atoms->size(); ia++) {
    auto pos1 = (*atoms)[ia].pos;
    pos1.x -= current_center[0];
    pos1.y -= current_center[1];
    pos1.z -= current_center[2];
    const auto pos2 = cached_ref_pos.col(ia);
    m_C(0, 0) += pos1.x * pos2[0];
    m_C(0, 1) += pos1.x * pos2[1];
    m_C(0, 2) += pos1.x * pos2[2];
    m_C(1, 0) += pos1.y * pos2[0];
    m_C(1, 1) += pos1.y * pos2[1];
    m_C(1, 2) += pos1.y * pos2[2];
    m_C(2, 0) += pos1.z * pos2[0];
    m_C(2, 1) += pos1.z * pos2[1];
    m_C(2, 2) += pos1.z * pos2[2];
    m_sum_square += pos1.x * pos1.x + pos1.y * pos1.y + pos1.z * pos1.z + pos2[0] * pos2[0] + pos2[1] * pos2[1] + pos2[2] * pos2[2];
  }
  m_S(0, 0) = m_C(0, 0) + m_C(1, 1) + m_C(2, 2);
  m_S(1, 0) = m_C(1, 2) - m_C(2, 1);
  m_S(0, 1) = m_S(1, 0);
  m_S(2, 0) =  -m_C(0, 2) + m_C(2, 0);
  m_S(0, 2) = m_S(2, 0);
  m_S(3, 0) = m_C(0, 1) - m_C(1, 0);
  m_S(0, 3) = m_S(3, 0);
  m_S(1, 1) = m_C(0, 0) - m_C(1, 1) - m_C(2, 2);
  m_S(2, 1) = m_C(0, 1) + m_C(1, 0);
  m_S(1, 2) = m_S(2, 1);
  m_S(3, 1) = m_C(0, 2) + m_C(2, 0);
  m_S(1, 3) = m_S(3, 1);
  m_S(2, 2) = -m_C(0, 0) + m_C(1, 1) - m_C(2, 2);
  m_S(3, 2) = m_C(1, 2) + m_C(2, 1);
  m_S(2, 3) = m_S(3, 2);
  m_S(3, 3) = - m_C(0, 0) - m_C(1, 1) + m_C(2, 2);
  m_eigensolver.compute(m_S);
  // eigenvalues are sorted in increasing order
  p->x.real_value = cvm::sqrt((m_sum_square - 2.0 * m_eigensolver.eigenvalues()(3)) / atoms->size());
}

void colvar::eigen_rmsd::eigen_rmsd_impl_::calc_gradients(colvar::eigen_rmsd* p) {
  const auto current_rmsd = p->x.real_value;
  auto factor = 1.0 / (atoms->size() * current_rmsd);
  if (!std::isfinite(factor)) factor = 0.0;
  // build the rotation matrix
  const auto& q = m_eigensolver.eigenvectors().col(3);
  m_rotation_matrix(0, 0) = q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3];
  m_rotation_matrix(0, 1) = 2.0 * (q[1] * q[2] - q[0] * q[3]);
  m_rotation_matrix(0, 2) = 2.0 * (q[1] * q[3] + q[0] * q[2]);
  m_rotation_matrix(1, 0) = 2.0 * (q[1] * q[2] + q[0] * q[3]);
  m_rotation_matrix(1, 1) = q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3];
  m_rotation_matrix(1, 2) = 2.0 * (q[2] * q[3] - q[0] * q[1]);
  m_rotation_matrix(2, 0) = 2.0 * (q[1] * q[3] - q[0] * q[2]);
  m_rotation_matrix(2, 1) = 2.0 * (q[2] * q[3] + q[0] * q[1]);
  m_rotation_matrix(2, 2) = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3];
  for (size_t ia = 0; ia < atoms->size(); ia++) {
    auto pos1 = (*atoms)[ia].pos;
    pos1.x -= current_center[0];
    pos1.y -= current_center[1];
    pos1.z -= current_center[2];
    const auto ref_rotated = m_rotation_matrix.transpose() * cached_ref_pos.col(ia);
    (*atoms)[ia].grad.x = factor * (pos1.x - ref_rotated[0]);
    (*atoms)[ia].grad.y = factor * (pos1.y - ref_rotated[1]);
    (*atoms)[ia].grad.z = factor * (pos1.z - ref_rotated[2]);
  }
}

void colvar::eigen_rmsd::calc_value() {
  p_impl->calc_value(this);
}

void colvar::eigen_rmsd::calc_gradients() {
  p_impl->calc_gradients(this);
}

void colvar::eigen_rmsd::apply_force(colvarvalue const &force)
{
  if (!(p_impl->atoms->noforce))
    p_impl->atoms->apply_colvar_force(force.real_value);
}
