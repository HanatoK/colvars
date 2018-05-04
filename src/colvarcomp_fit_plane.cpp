#include "colvarmodule.h"
#include "colvarvalue.h"
#include "colvarparse.h"
#include "colvar.h"
#include "colvarcomp.h"

colvar::fit_plane::fit_plane(std::string const &conf) : cvc(conf), groups(0) {
    function_type = "fit_plane";
    /// Check all available atom groups
    /// Assume groups exist first and looping
    bool has_groups = true;
    size_t group_index = 1;
    while (has_groups) {
        std::string group_name = "group" + std::to_string(group_index);
        /// Check the existence the group's name
        if (key_lookup(conf, group_name.c_str())) {
            groups.push_back(parse_group(conf, group_name.c_str()));
            ++group_index;
        } else {
            has_groups = false;
        }
    }
    if (groups.size() < 3) {
        cvm::error("Error: No enough atom groups(at least 3)!", INPUT_ERROR);
    }
    n = groups.size();
    /// x denotes the value of collective variable!
    x.type(colvarvalue::type_3vector);
    // I don't know what implicit gradient is, but other vector-based variables use it. So do I.
    enable(f_cvc_implicit_gradient);
    enable(f_cvc_com_based);
}

colvar::fit_plane::fit_plane() {
    function_type = "fit_plane";
    x.type(colvarvalue::type_3vector);
    enable(f_cvc_implicit_gradient);
    enable(f_cvc_com_based);
}

void colvar::fit_plane::calc_value() {
    // Reset all variables
    sum_xi = 0;
    sum_yi = 0;
    sum_zi = 0;
    sum_xi_yi = 0;
    sum_xi_zi = 0;
    sum_yi_zi = 0;
    sum_xi_square = 0;
    sum_yi_square = 0;
    g = 0;
    // TODO: PBC-aware COM calculation
    for (auto it_groups = groups.begin(); it_groups != groups.end(); ++it_groups) {
        // Calculate the COM of atom groups
        (*it_groups)->calc_center_of_mass();
        cvm::atom_pos group_pos = (*it_groups)->center_of_mass();
        // DEBUG logging
//         cvm::log(std::string("Fit plane: ") + std::to_string(group_pos.x) + " " + std::to_string(group_pos.y) + " " + std::to_string(group_pos.z) + '\n');
        // Sum up the necessary variables
        sum_xi += group_pos.x;
        sum_yi += group_pos.y;
        sum_zi += group_pos.z;
        sum_xi_yi += group_pos.x * group_pos.y;
        sum_xi_zi += group_pos.x * group_pos.z;
        sum_yi_zi += group_pos.y * group_pos.z;
        sum_xi_square += group_pos.x * group_pos.x;
        sum_yi_square += group_pos.y * group_pos.y;
    }
    g = sum_xi_square * (n * sum_yi_square - sum_yi * sum_yi) - sum_xi_yi * (n * sum_xi_yi - sum_xi * sum_yi) + sum_xi * (sum_xi_yi * sum_yi - sum_xi * sum_yi_square);
    if (g == 0) {
        cvm::error("Error: The plane may go through the z-axis!", FATAL_ERROR);
    }
    k0 = (sum_xi_zi * (sum_yi * sum_yi - n * sum_yi_square) + sum_yi_zi * (n * sum_xi_yi - sum_xi * sum_yi) + sum_zi * (sum_xi * sum_yi_square - sum_yi * sum_xi_yi)) / g;
    k1 = (sum_xi_zi * (n * sum_xi_yi - sum_xi * sum_yi) + sum_yi_zi * (sum_xi * sum_xi - n * sum_xi_square) + sum_zi * (sum_xi_square * sum_yi - sum_xi * sum_xi_yi)) / g;
    k2 = 1.0;
    norm = 1.0 / std::sqrt(k0 * k0 + k1 * k1 + 1);
    x[0] = k0 * norm;
    x[1] = k1 * norm;
    x[2] = k2 * norm;
    calc_gradients();
//     cvm::log("Fit plane: ======================================");
}

void colvar::fit_plane::calc_gradients() {
    dk0norm.assign(n, std::vector<double>(3, 0));
    dk1norm.assign(n, std::vector<double>(3, 0));
    dk2norm.assign(n, std::vector<double>(3, 0));
    const double f1 = sum_xi_zi * (sum_yi * sum_yi - n * sum_yi_square) + sum_yi_zi * (n * sum_xi_yi - sum_xi * sum_yi) + sum_zi * (sum_xi * sum_yi_square - sum_yi * sum_xi_yi);
    const double f2 = sum_xi_zi * (n * sum_xi_yi - sum_xi * sum_yi) + sum_yi_zi * (sum_xi * sum_xi - n * sum_xi_square) + sum_zi * (sum_xi_square * sum_yi - sum_xi * sum_xi_yi);
    for (size_t i = 0; i < n; ++i) {
        cvm::atom_pos group_pos = groups[i]->center_of_mass();
        const double xi = group_pos.x;
        const double yi = group_pos.y;
        const double zi = group_pos.z;
        const double dg_dxi = 2 * xi * (n * sum_yi_square - sum_yi * sum_yi) - yi * (n * sum_xi_yi - sum_xi * sum_yi) - sum_xi_yi * (n * yi - sum_yi) + sum_xi_yi * sum_yi - sum_xi * sum_yi_square + yi * sum_xi * sum_yi - sum_xi * sum_yi_square;
        const double dg_dyi = 2 * n * yi * sum_xi_square - 2 * sum_xi_square * sum_yi - n * xi * sum_xi_yi + xi * sum_xi * sum_yi - n * xi * sum_xi_yi + sum_xi * sum_xi_yi + xi * sum_xi * sum_yi + sum_xi * sum_xi_yi - 2 * yi * sum_xi * sum_xi;
        const double df1_dxi = zi * (sum_yi * sum_yi - n * sum_yi_square) + sum_yi_zi * (n * yi - sum_yi) + sum_zi * (sum_yi_square - yi * sum_yi);
        const double df1_dyi = sum_xi_zi * (2 * sum_yi - 2 * n * yi) + zi * (n * sum_xi_yi - sum_xi * sum_yi) + sum_yi_zi * (n * xi - sum_xi) + sum_zi * (2 * yi * sum_xi - (sum_xi_yi + xi * sum_yi));
        const double df1_dzi = xi * (sum_yi * sum_yi - n * sum_yi_square) + yi * (n * sum_xi_yi - sum_xi * sum_yi) + (sum_xi * sum_yi_square - sum_yi * sum_xi_yi);
        const double dk0_dxi = (df1_dxi * g - dg_dxi * f1) / (g * g);
        const double dk0_dyi = (df1_dyi * g - dg_dyi * f1) / (g * g);
        const double dk0_dzi = df1_dzi / g;
        const double df2_dxi = zi * (n * sum_xi_yi - sum_xi * sum_yi) + sum_xi_zi * (n * yi - sum_yi) + sum_yi_zi * (2 * sum_xi - 2 * n *xi) + sum_zi * (2 * xi * sum_yi - (sum_xi_yi + yi * sum_xi));
        const double df2_dyi = sum_xi_zi * (n * xi - sum_xi) + zi * (sum_xi * sum_xi - n * sum_xi_square) + sum_zi * (sum_xi_square - xi * sum_xi);
        const double df2_dzi = xi * (n * sum_xi_yi - sum_xi * sum_yi) + yi * (sum_xi * sum_xi - n * sum_xi_square) + (sum_xi_square * sum_yi - sum_xi * sum_xi_yi);
        const double dk1_dxi = (df2_dxi * g - dg_dxi * f2) / (g * g);
        const double dk1_dyi = (df2_dyi * g - dg_dyi * f2) / (g * g);
        const double dk1_dzi = df2_dzi / g;
        const double dnorm_dxi = 0.5 * norm * (2 * k0 * dk0_dxi + 2 * k1 * dk1_dxi) * (-1.0 / (k0 * k0 + k1 * k1 + 1));
        const double dnorm_dyi = 0.5 * norm * (2 * k0 * dk0_dyi + 2 * k1 * dk1_dyi) * (-1.0 / (k0 * k0 + k1 * k1 + 1));
        const double dnorm_dzi = 0.5 * norm * (2 * k0 * dk0_dzi + 2 * k1 * dk1_dzi) * (-1.0 / (k0 * k0 + k1 * k1 + 1));
        dk0norm[i][0] = dk0_dxi * norm + k0 * dnorm_dxi;
        dk0norm[i][1] = dk0_dyi * norm + k0 * dnorm_dyi;
        dk0norm[i][2] = dk0_dzi * norm + k0 * dnorm_dzi;
        dk1norm[i][0] = dk1_dxi * norm + k1 * dnorm_dxi;
        dk1norm[i][1] = dk1_dyi * norm + k1 * dnorm_dyi;
        dk1norm[i][2] = dk1_dzi * norm + k1 * dnorm_dzi;
        dk2norm[i][0] = 1.0 * dnorm_dxi;
        dk2norm[i][1] = 1.0 * dnorm_dyi;
        dk2norm[i][2] = 1.0 * dnorm_dzi;
        // DEBUG logging
//         cvm::log(std::string("Fit plane derivative: dk0/dx") + std::to_string(i+1) + std::string(": ") + std::to_string(dk0norm[i][0]) + '\n');
//         cvm::log(std::string("Fit plane derivative: dk0/dy") + std::to_string(i+1) + std::string(": ") + std::to_string(dk0norm[i][1]) + '\n');
//         cvm::log(std::string("Fit plane derivative: dk0/dz") + std::to_string(i+1) + std::string(": ") + std::to_string(dk0norm[i][2]) + '\n');
//         cvm::log(std::string("Fit plane derivative: dk1/dx") + std::to_string(i+1) + std::string(": ") + std::to_string(dk1norm[i][0]) + '\n');
//         cvm::log(std::string("Fit plane derivative: dk1/dy") + std::to_string(i+1) + std::string(": ") + std::to_string(dk1norm[i][1]) + '\n');
//         cvm::log(std::string("Fit plane derivative: dk1/dz") + std::to_string(i+1) + std::string(": ") + std::to_string(dk1norm[i][2]) + '\n');
//         cvm::log(std::string("Fit plane derivative: dk2/dx") + std::to_string(i+1) + std::string(": ") + std::to_string(dk2norm[i][0]) + '\n');
//         cvm::log(std::string("Fit plane derivative: dk2/dy") + std::to_string(i+1) + std::string(": ") + std::to_string(dk2norm[i][1]) + '\n');
//         cvm::log(std::string("Fit plane derivative: dk2/dz") + std::to_string(i+1) + std::string(": ") + std::to_string(dk2norm[i][2]) + '\n');
    }
}

void colvar::fit_plane::apply_force(colvarvalue const &force) {
    cvm::rvector const &Fr = force.rvector_value;
    cvm::rvector tmp_group_force;
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            tmp_group_force[j] = Fr[0] * dk0norm[i][j] + Fr[1] * dk1norm[i][j] + Fr[2] * dk2norm[i][j];
        }
        groups[i]->apply_force(tmp_group_force);
    }
} 
