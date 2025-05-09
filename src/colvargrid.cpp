// -*- c++ -*-

// This file is part of the Collective Variables module (Colvars).
// The original version of Colvars and its updates are located at:
// https://github.com/Colvars/colvars
// Please update all Colvars source files before making any changes.
// If you wish to distribute your changes, please submit them to the
// Colvars repository at GitHub.

#include <ctime>

#include "colvarmodule.h"
#include "colvarvalue.h"
#include "colvarparse.h"
#include "colvar.h"
#include "colvargrid.h"
#include "colvargrid_def.h"



colvar_grid_count::colvar_grid_count()
  : colvar_grid<size_t>()
{
  mult = 1;
}

colvar_grid_count::colvar_grid_count(std::vector<colvar *>  &colvars,
                                     std::string config)
  : colvar_grid<size_t>(colvars, 0, 1, false, nullptr, config)
{}

colvar_grid_count::colvar_grid_count(std::vector<colvar *>  &colvars,
                                     std::shared_ptr<const colvar_grid_params> params)
  : colvar_grid<size_t>(colvars, 0, 1, false, params)
{}

std::string colvar_grid_count::get_state_params() const
{
  return colvar_grid<size_t>::get_state_params();
}

int colvar_grid_count::parse_params(std::string const &conf,
                                    colvarparse::Parse_Mode const parse_mode)
{
  return colvar_grid<size_t>::parse_params(conf, parse_mode);
}

std::istream & colvar_grid_count::read_restart(std::istream &is)
{
  return colvar_grid<size_t>::read_restart(is);
}

cvm::memory_stream & colvar_grid_count::read_restart(cvm::memory_stream &is)
{
  return colvar_grid<size_t>::read_restart(is);
}

std::ostream & colvar_grid_count::write_restart(std::ostream &os)
{
  return colvar_grid<size_t>::write_restart(os);
}

cvm::memory_stream & colvar_grid_count::write_restart(cvm::memory_stream &os)
{
  return colvar_grid<size_t>::write_restart(os);
}

std::istream &colvar_grid_count::read_raw(std::istream &is)
{
  return colvar_grid<size_t>::read_raw(is);
}

cvm::memory_stream &colvar_grid_count::read_raw(cvm::memory_stream &is)
{
  return colvar_grid<size_t>::read_raw(is);
}

std::ostream &colvar_grid_count::write_raw(std::ostream &os, size_t const buf_size) const
{
  return colvar_grid<size_t>::write_raw(os, buf_size);
}

cvm::memory_stream &colvar_grid_count::write_raw(cvm::memory_stream &os,
                                                 size_t const buf_size) const
{
  return colvar_grid<size_t>::write_raw(os, buf_size);
}

std::istream & colvar_grid_count::read_multicol(std::istream &is, bool add)
{
  return colvar_grid<size_t>::read_multicol(is, add);
}

int colvar_grid_count::read_multicol(std::string const &filename,
                                     std::string description,
                                     bool add)
{
  return colvar_grid<size_t>::read_multicol(filename, description, add);
}

std::ostream & colvar_grid_count::write_multicol(std::ostream &os) const
{
  return colvar_grid<size_t>::write_multicol(os);
}

int colvar_grid_count::write_multicol(std::string const &filename,
                                      std::string description) const
{
  return colvar_grid<size_t>::write_multicol(filename, description);
}

std::ostream & colvar_grid_count::write_opendx(std::ostream &os) const
{
  return colvar_grid<size_t>::write_opendx(os);
}

int colvar_grid_count::write_opendx(std::string const &filename,
                                    std::string description) const
{
  return colvar_grid<size_t>::write_opendx(filename, description);
}



colvar_grid_scalar::colvar_grid_scalar()
  : colvar_grid<cvm::real>(), samples(NULL)
{}

colvar_grid_scalar::colvar_grid_scalar(colvar_grid_scalar const &g)
  : colvar_grid<cvm::real>(g), samples(NULL)
{
}

colvar_grid_scalar::colvar_grid_scalar(std::vector<colvar *> &colvars,
                                       std::shared_ptr<const colvar_grid_params> params,
                                       bool add_extra_bin,
                                       std::string config)
  : colvar_grid<cvm::real>(colvars, 0.0, 1, add_extra_bin, params, config), samples(NULL)
{
}

colvar_grid_scalar::colvar_grid_scalar(std::string const &filename)
  : colvar_grid<cvm::real>(filename, 1),
    samples(nullptr)
{
}

colvar_grid_scalar::~colvar_grid_scalar()
{
}

std::string colvar_grid_scalar::get_state_params() const
{
  return colvar_grid<cvm::real>::get_state_params();
}

int colvar_grid_scalar::parse_params(std::string const &conf,
                                    colvarparse::Parse_Mode const parse_mode)
{
  return colvar_grid<cvm::real>::parse_params(conf, parse_mode);
}

std::istream &colvar_grid_scalar::read_restart(std::istream &is)
{
  return colvar_grid<cvm::real>::read_restart(is);
}

cvm::memory_stream &colvar_grid_scalar::read_restart(cvm::memory_stream &is)
{
  return colvar_grid<cvm::real>::read_restart(is);
}

std::ostream &colvar_grid_scalar::write_restart(std::ostream &os)
{
  return colvar_grid<cvm::real>::write_restart(os);
}

cvm::memory_stream &colvar_grid_scalar::write_restart(cvm::memory_stream &os)
{
  return colvar_grid<cvm::real>::write_restart(os);
}

std::istream &colvar_grid_scalar::read_raw(std::istream &is)
{
  return colvar_grid<cvm::real>::read_raw(is);
}

cvm::memory_stream &colvar_grid_scalar::read_raw(cvm::memory_stream &is)
{
  return colvar_grid<cvm::real>::read_raw(is);
}

std::ostream &colvar_grid_scalar::write_raw(std::ostream &os, size_t const buf_size) const
{
  return colvar_grid<cvm::real>::write_raw(os, buf_size);
}

cvm::memory_stream &colvar_grid_scalar::write_raw(cvm::memory_stream &os,
                                                  size_t const buf_size) const
{
  return colvar_grid<cvm::real>::write_raw(os, buf_size);
}

std::istream & colvar_grid_scalar::read_multicol(std::istream &is, bool add)
{
  return colvar_grid<cvm::real>::read_multicol(is, add);
}

int colvar_grid_scalar::read_multicol(std::string const &filename,
                                      std::string description,
                                      bool add)
{
  return colvar_grid<cvm::real>::read_multicol(filename, description, add);
}

std::ostream & colvar_grid_scalar::write_multicol(std::ostream &os) const
{
  return colvar_grid<cvm::real>::write_multicol(os);
}

int colvar_grid_scalar::write_multicol(std::string const &filename,
                                       std::string description) const
{
  return colvar_grid<cvm::real>::write_multicol(filename, description);
}

std::ostream & colvar_grid_scalar::write_opendx(std::ostream &os) const
{
  return colvar_grid<cvm::real>::write_opendx(os);
}

int colvar_grid_scalar::write_opendx(std::string const &filename,
                                     std::string description) const
{
  return colvar_grid<cvm::real>::write_opendx(filename, description);
}


cvm::real colvar_grid_scalar::maximum_value() const
{
  cvm::real max = data[0];
  for (size_t i = 0; i < nt; i++) {
    if (data[i] > max) max = data[i];
  }
  return max;
}


cvm::real colvar_grid_scalar::minimum_value() const
{
  cvm::real min = data[0];
  for (size_t i = 0; i < nt; i++) {
    if (data[i] < min) min = data[i];
  }
  return min;
}

cvm::real colvar_grid_scalar::minimum_pos_value() const
{
  cvm::real minpos = data[0];
  size_t i;
  for (i = 0; i < nt; i++) {
    if(data[i] > 0) {
      minpos = data[i];
      break;
    }
  }
  for (i = 0; i < nt; i++) {
    if (data[i] > 0 && data[i] < minpos) minpos = data[i];
  }
  return minpos;
}

cvm::real colvar_grid_scalar::integral() const
{
  cvm::real sum = 0.0;
  for (size_t i = 0; i < nt; i++) {
    sum += data[i];
  }
  cvm::real bin_volume = 1.0;
  for (size_t id = 0; id < widths.size(); id++) {
    bin_volume *= widths[id];
  }
  return bin_volume * sum;
}


cvm::real colvar_grid_scalar::entropy() const
{
  cvm::real sum = 0.0;
  for (size_t i = 0; i < nt; i++) {
    if (data[i] >0) {
      sum += -1.0 * data[i] * cvm::logn(data[i]);
    }
  }
  cvm::real bin_volume = 1.0;
  for (size_t id = 0; id < widths.size(); id++) {
    bin_volume *= widths[id];
  }
  return bin_volume * sum;
}

/// \brief Return the RMSD between this grid's data and another one
/// Grids must have the same dimensions.
cvm::real colvar_grid_scalar::grid_rmsd(colvar_grid_scalar const &other_grid) const
{
  if (other_grid.data.size() != this->data.size()) {
    cvm::error("Error: trying to subtract two grids with "
                "different size.\n");
    return -1.;
  }

  cvm::real sum2 = 0.0;

  if (samples && other_grid.samples) {
    for (size_t i = 0; i < data.size(); i++) {
      size_t n = samples->get_value(i);
      cvm::real us = n ? data[i] / n : 0.0;
      n = other_grid.samples->get_value(i);
      cvm::real them = n ? other_grid.data[i ] / n : 0.0;
      cvm::real d = us - them;
      sum2 += d*d;
    }
  } else {
    for (size_t i = 0; i < data.size(); i++) {
      cvm::real d = other_grid.data[i] - data[i];
      sum2 += d*d;
    }
  }

  return sqrt(sum2/this->data.size());
}


colvar_grid_gradient::colvar_grid_gradient()
  : colvar_grid<cvm::real>(), samples(NULL)
{}


// colvar_grid_gradient::colvar_grid_gradient(std::vector<colvar *> &colvars, std::string config)
//   : colvar_grid<cvm::real>(colvars, 0.0, colvars.size(), false, nullptr, config), samples(NULL)
// {}

// colvar_grid_gradient::colvar_grid_gradient(std::vector<colvar *> &colvars,
//                                            std::shared_ptr<colvar_grid_count> samples_in)
//   : colvar_grid<cvm::real>(colvars, 0.0, colvars.size(), false, samples_in), samples(samples_in)
// {
//   if (samples_in)
//     samples_in->has_parent_data = true;
// }

colvar_grid_gradient::colvar_grid_gradient(std::vector<colvar *> &colvars,
                                           std::shared_ptr<colvar_grid_count> samples_in,
                                           std::shared_ptr<const colvar_grid_params> params,
                                           std::string config)
  : colvar_grid<cvm::real>(colvars, 0.0, colvars.size(), false, params, config), samples(samples_in)
{
  if (samples_in)
    samples_in->has_parent_data = true;
}


colvar_grid_gradient::colvar_grid_gradient(std::string const &filename)
  : colvar_grid<cvm::real>(filename, 0),
    samples(nullptr)
{
}

std::string colvar_grid_gradient::get_state_params() const
{
  return colvar_grid<cvm::real>::get_state_params();
}

int colvar_grid_gradient::parse_params(std::string const &conf,
                                       colvarparse::Parse_Mode const parse_mode)
{
  return colvar_grid<cvm::real>::parse_params(conf, parse_mode);
}

std::istream &colvar_grid_gradient::read_restart(std::istream &is)
{
  return colvar_grid<cvm::real>::read_restart(is);
}

cvm::memory_stream &colvar_grid_gradient::read_restart(cvm::memory_stream &is)
{
  return colvar_grid<cvm::real>::read_restart(is);
}

std::ostream &colvar_grid_gradient::write_restart(std::ostream &os)
{
  return colvar_grid<cvm::real>::write_restart(os);
}

cvm::memory_stream &colvar_grid_gradient::write_restart(cvm::memory_stream &os)
{
  return colvar_grid<cvm::real>::write_restart(os);
}

std::istream &colvar_grid_gradient::read_raw(std::istream &is)
{
  return colvar_grid<cvm::real>::read_raw(is);
}

cvm::memory_stream &colvar_grid_gradient::read_raw(cvm::memory_stream &is)
{
  return colvar_grid<cvm::real>::read_raw(is);
}

std::ostream &colvar_grid_gradient::write_raw(std::ostream &os, size_t const buf_size) const
{
  return colvar_grid<cvm::real>::write_raw(os, buf_size);
}

cvm::memory_stream &colvar_grid_gradient::write_raw(cvm::memory_stream &os,
                                                    size_t const buf_size) const
{
  return colvar_grid<cvm::real>::write_raw(os, buf_size);
}

std::istream & colvar_grid_gradient::read_multicol(std::istream &is, bool add)
{
  return colvar_grid<cvm::real>::read_multicol(is, add);
}

int colvar_grid_gradient::read_multicol(std::string const &filename,
                                        std::string description,
                                        bool add)
{
  return colvar_grid<cvm::real>::read_multicol(filename, description, add);
}

std::ostream & colvar_grid_gradient::write_multicol(std::ostream &os) const
{
  return colvar_grid<cvm::real>::write_multicol(os);
}

int colvar_grid_gradient::write_multicol(std::string const &filename,
                                         std::string description) const
{
  return colvar_grid<cvm::real>::write_multicol(filename, description);
}

std::ostream & colvar_grid_gradient::write_opendx(std::ostream &os) const
{
  return colvar_grid<cvm::real>::write_opendx(os);
}

int colvar_grid_gradient::write_opendx(std::string const &filename,
                                       std::string description) const
{
  return colvar_grid<cvm::real>::write_opendx(filename, description);
}


void colvar_grid_gradient::write_1D_integral(std::ostream &os)
{
  cvm::real bin, min, integral;
  std::vector<cvm::real> int_vals;

  os << "#       xi            A(xi)\n";

  if (cv.size() != 1) {
    cvm::error("Cannot write integral for multi-dimensional gradient grids.");
    return;
  }

  integral = 0.0;
  int_vals.push_back(0.0);
  min = 0.0;

  // correction for periodic colvars, so that the PMF is periodic
  cvm::real corr;
  if (periodic[0]) {
    corr = average();
  } else {
    corr = 0.0;
  }

  for (std::vector<int> ix = new_index(); index_ok(ix); incr(ix)) {

    if (samples) {
      size_t const samples_here = samples->value(ix);
      if (samples_here)
        integral += (value(ix) / cvm::real(samples_here) - corr) * cv[0]->width;
    } else {
      integral += (value(ix) - corr) * cv[0]->width;
    }

    if ( integral < min ) min = integral;
    int_vals.push_back(integral);
  }

  bin = 0.0;
  for ( int i = 0; i < nx[0]; i++, bin += 1.0 ) {
    os << std::setw(10) << cv[0]->lower_boundary.real_value + cv[0]->width * bin << " "
       << std::setw(cvm::cv_width)
       << std::setprecision(cvm::cv_prec)
       << int_vals[i] - min << "\n";
  }

  os << std::setw(10) << cv[0]->lower_boundary.real_value + cv[0]->width * bin << " "
     << std::setw(cvm::cv_width)
     << std::setprecision(cvm::cv_prec)
     << int_vals[nx[0]] - min << "\n";

  return;
}


/// \brief Return the RMSD between this grid's data and another one
/// Grids must have the same dimensions.
cvm::real colvar_grid_gradient::grid_rmsd(colvar_grid_gradient const &other_grid) const
{
  if (other_grid.multiplicity() != this->multiplicity()) {
    cvm::error("Error: trying to subtract two grids with "
                "different multiplicity.\n");
    return -1.;
  }

  if (other_grid.data.size() != this->data.size()) {
    cvm::error("Error: trying to subtract two grids with "
                "different size.\n");
    return -1.;
  }

  cvm::real sum2 = 0.0;
  std::vector<int> ix;
  size_t imult;
  for (ix = new_index(); index_ok(ix); incr(ix)) {
    for (imult = 0; imult < this->multiplicity(); imult++) {
      cvm::real d = this->value_output(ix, imult) - other_grid.value_output(ix, imult);
      sum2 += d*d;
    }
  }
  return sqrt(sum2/this->data.size());
}


integrate_potential::integrate_potential(std::vector<colvar *> &colvars,
                                         std::shared_ptr<colvar_grid_gradient> gradients)
  : colvar_grid_scalar(colvars, gradients, true),
    b_smoothed(false),
    gradients(gradients)
{
  // parent class colvar_grid_scalar is constructed with add_extra_bin option set to true
  // hence PMF grid is wider than gradient grid if non-PBC

  if (nd > 1) {
    cvm::main()->cite_feature("Poisson integration of 2D/3D free energy surfaces");
    divergence.resize(nt);

    // Compute inverse of Laplacian diagonal for Jacobi preconditioning
    // For now all code related to preconditioning is commented out
    // until a method better than Jacobi is implemented
//     cvm::log("Preparing inverse diagonal for preconditioning...\n");
//     inv_lap_diag.resize(nt);
//     std::vector<cvm::real> id(nt), lap_col(nt);
//     for (int i = 0; i < nt; i++) {
//       if (i % (nt / 100) == 0)
//         cvm::log(cvm::to_str(i));
//       id[i] = 1.;
//       atimes(id, lap_col);
//       id[i] = 0.;
//       inv_lap_diag[i] = 1. / lap_col[i];
//     }
//     cvm::log("Done.\n");
  }
}


integrate_potential::integrate_potential(std::shared_ptr<colvar_grid_gradient> gradients)
  : b_smoothed(false),
    gradients(gradients)
{
  nd = gradients->num_variables();
  nx = gradients->number_of_points_vec();
  widths = gradients->widths;
  periodic = gradients->periodic;

  // Expand grid by 1 bin in non-periodic dimensions
  for (size_t i = 0; i < nd; i++ ) {
    if (!periodic[i]) nx[i]++;
    // Shift the grid by half the bin width (values at edges instead of center of bins)
    lower_boundaries.push_back(gradients->lower_boundaries[i].real_value - 0.5 * widths[i]);
  }

  setup(nx);

  if (nd > 1) {
    divergence.resize(nt);
  }
}


int integrate_potential::integrate(const int itmax, const cvm::real &tol, cvm::real & err, bool verbose)
{
  int iter = 0;

  if (nd == 1) {

    cvm::real sum = 0.0;
    cvm::real corr;
    if ( periodic[0] ) {
      corr = gradients->average(); // Enforce PBC by subtracting average gradient
    } else {
      corr = 0.0;
    }
    std::vector<int> ix;
    // Iterate over valid indices in gradient grid
    for (ix = new_index(); gradients->index_ok(ix); incr(ix)) {
      set_value(ix, sum);
      cvm::real val = gradients->value_output_smoothed(ix, b_smoothed);
      sum += (val - corr) * widths[0];
    }
    if (index_ok(ix)) {
      // This will happen if non-periodic: then PMF grid has one extra bin wrt gradient grid
      // If not, sum should be zero
      set_value(ix, sum);
    }

  } else if (nd <= 3) {

    nr_linbcg_sym(divergence, data, tol, itmax, iter, err);
    if (verbose)
      cvm::log("Integrated in " + cvm::to_str(iter) + " steps, error: " + cvm::to_str(err));

  } else {
    cvm::error("Cannot integrate PMF in dimension > 3\n");
  }

  return iter;
}


void integrate_potential::set_div()
{
  if (nd == 1) return;
  for (std::vector<int> ix = new_index(); index_ok(ix); incr(ix)) {
    update_div_local(ix);
  }
}


void integrate_potential::update_div_neighbors(const std::vector<int> &ix0)
{
  std::vector<int> ix(ix0);
  int i, j, k;

  // If not periodic, expanded grid ensures that upper neighbors of ix0 are valid grid points
  if (nd == 1) {
    return;

  } else if (nd == 2) {

    update_div_local(ix);
    ix[0]++; wrap(ix);
    update_div_local(ix);
    ix[1]++; wrap(ix);
    update_div_local(ix);
    ix[0]--; wrap(ix);
    update_div_local(ix);

  } else if (nd == 3) {

    for (i = 0; i<2; i++) {
      ix[1] = ix0[1];
      for (j = 0; j<2; j++) {
        ix[2] = ix0[2];
        for (k = 0; k<2; k++) {
          wrap(ix);
          update_div_local(ix);
          ix[2]++;
        }
        ix[1]++;
      }
      ix[0]++;
    }
  }
}


void integrate_potential::get_grad(cvm::real * g, std::vector<int> &ix)
{
  size_t i;
  bool edge = gradients->wrap_detect_edge(ix); // Detect edge if non-PBC

  if (edge) {
    for ( i = 0; i<nd; i++ ) {
          g[i] = 0.0;
    }
    return;
  }

  gradients->vector_value_smoothed(ix, g, b_smoothed);
}


void integrate_potential::update_div_local(const std::vector<int> &ix0)
{
  const size_t linear_index = address(ix0);
  int i, j, k;
  std::vector<int> ix = ix0;

  if (nd == 2) {
    // gradients at grid points surrounding the current scalar grid point
    cvm::real g00[2], g01[2], g10[2], g11[2];

    get_grad(g11, ix);
    ix[0] = ix0[0] - 1;
    get_grad(g01, ix);
    ix[1] = ix0[1] - 1;
    get_grad(g00, ix);
    ix[0] = ix0[0];
    get_grad(g10, ix);

    divergence[linear_index] = ((g10[0]-g00[0] + g11[0]-g01[0]) / widths[0]
                              + (g01[1]-g00[1] + g11[1]-g10[1]) / widths[1]) * 0.5;
  } else if (nd == 3) {
    cvm::real gc[24]; // stores 3d gradients in 8 contiguous bins
    int index = 0;

    ix[0] = ix0[0] - 1;
    for (i = 0; i<2; i++) {
      ix[1] = ix0[1] - 1;
      for (j = 0; j<2; j++) {
        ix[2] = ix0[2] - 1;
        for (k = 0; k<2; k++) {
          get_grad(gc + index, ix);
          index += 3;
          ix[2]++;
        }
        ix[1]++;
      }
      ix[0]++;
    }

    divergence[linear_index] =
     ((gc[3*4]-gc[0] + gc[3*5]-gc[3*1] + gc[3*6]-gc[3*2] + gc[3*7]-gc[3*3])
      / widths[0]
    + (gc[3*2+1]-gc[0+1] + gc[3*3+1]-gc[3*1+1] + gc[3*6+1]-gc[3*4+1] + gc[3*7+1]-gc[3*5+1])
      / widths[1]
    + (gc[3*1+2]-gc[0+2] + gc[3*3+2]-gc[3*2+2] + gc[3*5+2]-gc[3*4+2] + gc[3*7+2]-gc[3*6+2])
      / widths[2]) * 0.25;
  }
}


/// Multiplication by sparse matrix representing Laplacian
/// NOTE: Laplacian must be symmetric for solving with CG
void integrate_potential::atimes(const std::vector<cvm::real> &A, std::vector<cvm::real> &LA)
{
  if (nd == 2) {
    // DIMENSION 2

    size_t index, index2;
    int i, j;
    cvm::real fact;
    const cvm::real ffx = 1.0 / (widths[0] * widths[0]);
    const cvm::real ffy = 1.0 / (widths[1] * widths[1]);
    const int h = nx[1];
    const int w = nx[0];
    // offsets for 4 reference points of the Laplacian stencil
    int xm = -h;
    int xp =  h;
    int ym = -1;
    int yp =  1;

    // NOTE on performance: this version is slightly sub-optimal because
    // it contains two double loops on the core of the array (for x and y terms)
    // The slightly faster version is in commit 0254cb5a2958cb2e135f268371c4b45fad34866b
    // yet it is much uglier, and probably horrible to extend to dimension 3
    // All terms in the matrix are assigned (=) during the x loops, then updated (+=)
    // with the y (and z) contributions


    // All x components except on x edges
    index = h; // Skip first column

    // Halve the term on y edges (if any) to preserve symmetry of the Laplacian matrix
    // (Long Chen, Finite Difference Methods, UCI, 2017)
    fact = periodic[1] ? 1.0 : 0.5;

    for (i=1; i<w-1; i++) {
      // Full range of j, but factor may change on y edges (j == 0 and j == h-1)
      LA[index] = fact * ffx * (A[index + xm] + A[index + xp] - 2.0 * A[index]);
      index++;
      for (j=1; j<h-1; j++) {
        LA[index] = ffx * (A[index + xm] + A[index + xp] - 2.0 * A[index]);
        index++;
      }
      LA[index] = fact * ffx * (A[index + xm] + A[index + xp] - 2.0 * A[index]);
      index++;
    }
    // Edges along x (x components only)
    index = 0L; // Follows left edge
    index2 = h * static_cast<size_t>(w - 1); // Follows right edge
    if (periodic[0]) {
      xm =  h * (w - 1);
      xp =  h;
      fact = periodic[1] ? 1.0 : 0.5;
      LA[index]  = fact * ffx * (A[index + xm] + A[index + xp] - 2.0 * A[index]);
      LA[index2] = fact * ffx * (A[index2 - xp] + A[index2 - xm] - 2.0 * A[index2]);
      index++;
      index2++;
      for (j=1; j<h-1; j++) {
        LA[index]  = ffx * (A[index + xm] + A[index + xp] - 2.0 * A[index]);
        LA[index2] = ffx * (A[index2 - xp] + A[index2 - xm] - 2.0 * A[index2]);
        index++;
        index2++;
      }
      LA[index]  = fact * ffx * (A[index + xm] + A[index + xp] - 2.0 * A[index]);
      LA[index2] = fact * ffx * (A[index2 - xp] + A[index2 - xm] - 2.0 * A[index2]);
    } else {
      xm = -h;
      xp =  h;
      fact = periodic[1] ? 1.0 : 0.5; // Halve in corners in full PBC only
      // lower corner, "j == 0"
      LA[index]  = fact * ffx * (A[index + xp] - A[index]);
      LA[index2] = fact * ffx * (A[index2 + xm] - A[index2]);
      index++;
      index2++;
      for (j=1; j<h-1; j++) {
        // x gradient (+ y term of laplacian, calculated below)
        LA[index]  = ffx * (A[index + xp] - A[index]);
        LA[index2] = ffx * (A[index2 + xm] - A[index2]);
        index++;
        index2++;
      }
      // upper corner, j == h-1
      LA[index]  = fact * ffx * (A[index + xp] - A[index]);
      LA[index2] = fact * ffx * (A[index2 + xm] - A[index2]);
    }

    // Now adding all y components
    // All y components except on y edges
    index = 1; // Skip first element (in first row)

    fact = periodic[0] ? 1.0 : 0.5; // for i == 0
    for (i=0; i<w; i++) {
      // Factor of 1/2 on x edges if non-periodic
      if (i == 1) fact = 1.0;
      if (i == w - 1) fact = periodic[0] ? 1.0 : 0.5;
      for (j=1; j<h-1; j++) {
        LA[index] += fact * ffy * (A[index + ym] + A[index + yp] - 2.0 * A[index]);
        index++;
      }
      index += 2; // skip the edges and move to next column
    }
    // Edges along y (y components only)
    index = 0L; // Follows bottom edge
    index2 = h - 1; // Follows top edge
    if (periodic[1]) {
      fact = periodic[0] ? 1.0 : 0.5;
      ym = h - 1;
      yp = 1;
      LA[index]  += fact * ffy * (A[index + ym] + A[index + yp] - 2.0 * A[index]);
      LA[index2] += fact * ffy * (A[index2 - yp] + A[index2 - ym] - 2.0 * A[index2]);
      index  += h;
      index2 += h;
      for (i=1; i<w-1; i++) {
        LA[index]  += ffy * (A[index + ym] + A[index + yp] - 2.0 * A[index]);
        LA[index2] += ffy * (A[index2 - yp] + A[index2 - ym] - 2.0 * A[index2]);
        index  += h;
        index2 += h;
      }
      LA[index]  += fact * ffy * (A[index + ym] + A[index + yp] - 2.0 * A[index]);
      LA[index2] += fact * ffy * (A[index2 - yp] + A[index2 - ym] - 2.0 * A[index2]);
    } else {
      ym = -1;
      yp = 1;
      fact = periodic[0] ? 1.0 : 0.5; // Halve in corners in full PBC only
      // Left corner
      LA[index]  += fact * ffy * (A[index + yp] - A[index]);
      LA[index2] += fact * ffy * (A[index2 + ym] - A[index2]);
      index  += h;
      index2 += h;
      for (i=1; i<w-1; i++) {
        // y gradient (+ x term of laplacian, calculated above)
        LA[index]  += ffy * (A[index + yp] - A[index]);
        LA[index2] += ffy * (A[index2 + ym] - A[index2]);
        index  += h;
        index2 += h;
      }
      // Right corner
      LA[index]  += fact * ffy * (A[index + yp] - A[index]);
      LA[index2] += fact * ffy * (A[index2 + ym] - A[index2]);
    }

  } else if (nd == 3) {
    // DIMENSION 3

    int i, j, k;
    size_t index, index2;
    cvm::real fact = 1.0;
    const cvm::real ffx = 1.0 / (widths[0] * widths[0]);
    const cvm::real ffy = 1.0 / (widths[1] * widths[1]);
    const cvm::real ffz = 1.0 / (widths[2] * widths[2]);
    const int h = nx[2]; // height
    const int d = nx[1]; // depth
    const int w = nx[0]; // width
    // offsets for 6 reference points of the Laplacian stencil
    int xm = -d * h;
    int xp =  d * h;
    int ym = -h;
    int yp =  h;
    int zm = -1;
    int zp =  1;

    cvm::real factx = periodic[0] ? 1 : 0.5; // factor to be applied on x edges
    cvm::real facty = periodic[1] ? 1 : 0.5; // same for y
    cvm::real factz = periodic[2] ? 1 : 0.5; // same for z
    cvm::real ifactx = 1 / factx;
    cvm::real ifacty = 1 / facty;
    cvm::real ifactz = 1 / factz;

    // All x components except on x edges
    index = d * static_cast<size_t>(h); // Skip left slab
    fact = facty * factz;
    for (i=1; i<w-1; i++) {
      for (j=0; j<d; j++) { // full range of y
        if (j == 1) fact *= ifacty;
        if (j == d-1) fact *= facty;
        LA[index] = fact * ffx * (A[index + xm] + A[index + xp] - 2.0 * A[index]);
        index++;
        fact *= ifactz;
        for (k=1; k<h-1; k++) { // full range of z
          LA[index] = fact * ffx * (A[index + xm] + A[index + xp] - 2.0 * A[index]);
          index++;
        }
        fact *= factz;
        LA[index] = fact * ffx * (A[index + xm] + A[index + xp] - 2.0 * A[index]);
        index++;
      }
    }
    // Edges along x (x components only)
    index = 0L; // Follows left slab
    index2 = static_cast<size_t>(d) * h * (w - 1); // Follows right slab
    if (periodic[0]) {
      xm =  d * h * (w - 1);
      xp =  d * h;
      fact = facty * factz;
      for (j=0; j<d; j++) {
        if (j == 1) fact *= ifacty;
        if (j == d-1) fact *= facty;
        LA[index]  = fact * ffx * (A[index + xm] + A[index + xp] - 2.0 * A[index]);
        LA[index2] = fact * ffx * (A[index2 - xp] + A[index2 - xm] - 2.0 * A[index2]);
        index++;
        index2++;
        fact *= ifactz;
        for (k=1; k<h-1; k++) {
          LA[index]  = fact * ffx * (A[index + xm] + A[index + xp] - 2.0 * A[index]);
          LA[index2] = fact * ffx * (A[index2 - xp] + A[index2 - xm] - 2.0 * A[index2]);
          index++;
          index2++;
        }
        fact *= factz;
        LA[index]  = fact * ffx * (A[index + xm] + A[index + xp] - 2.0 * A[index]);
        LA[index2] = fact * ffx * (A[index2 - xp] + A[index2 - xm] - 2.0 * A[index2]);
        index++;
        index2++;
      }
    } else {
      xm = -d * h;
      xp =  d * h;
      fact = facty * factz;
      for (j=0; j<d; j++) {
        if (j == 1) fact *= ifacty;
        if (j == d-1) fact *= facty;
        LA[index]  = fact * ffx * (A[index + xp] - A[index]);
        LA[index2] = fact * ffx * (A[index2 + xm] - A[index2]);
        index++;
        index2++;
        fact *= ifactz;
        for (k=1; k<h-1; k++) {
          // x gradient (+ y, z terms of laplacian, calculated below)
          LA[index]  = fact * ffx * (A[index + xp] - A[index]);
          LA[index2] = fact * ffx * (A[index2 + xm] - A[index2]);
          index++;
          index2++;
        }
        fact *= factz;
        LA[index]  = fact * ffx * (A[index + xp] - A[index]);
        LA[index2] = fact * ffx * (A[index2 + xm] - A[index2]);
        index++;
        index2++;
      }
    }

    // Now adding all y components
    // All y components except on y edges
    index = h; // Skip first column (in front slab)
    fact = factx * factz;
    for (i=0; i<w; i++) { // full range of x
      if (i == 1) fact *= ifactx;
      if (i == w-1) fact *= factx;
      for (j=1; j<d-1; j++) {
        LA[index] += fact * ffy * (A[index + ym] + A[index + yp] - 2.0 * A[index]);
        index++;
        fact *= ifactz;
        for (k=1; k<h-1; k++) {
          LA[index] += fact * ffy * (A[index + ym] + A[index + yp] - 2.0 * A[index]);
          index++;
        }
        fact *= factz;
        LA[index] += fact * ffy * (A[index + ym] + A[index + yp] - 2.0 * A[index]);
        index++;
      }
      index += 2 * h; // skip columns in front and back slabs
    }
    // Edges along y (y components only)
    index = 0L; // Follows front slab
    index2 = h * static_cast<size_t>(d - 1); // Follows back slab
    if (periodic[1]) {
      ym = h * (d - 1);
      yp = h;
      fact = factx * factz;
      for (i=0; i<w; i++) {
        if (i == 1) fact *= ifactx;
        if (i == w-1) fact *= factx;
        LA[index]  += fact * ffy * (A[index + ym] + A[index + yp] - 2.0 * A[index]);
        LA[index2] += fact * ffy * (A[index2 - yp] + A[index2 - ym] - 2.0 * A[index2]);
        index++;
        index2++;
        fact *= ifactz;
        for (k=1; k<h-1; k++) {
          LA[index]  += fact * ffy * (A[index + ym] + A[index + yp] - 2.0 * A[index]);
          LA[index2] += fact * ffy * (A[index2 - yp] + A[index2 - ym] - 2.0 * A[index2]);
          index++;
          index2++;
        }
        fact *= factz;
        LA[index]  += fact * ffy * (A[index + ym] + A[index + yp] - 2.0 * A[index]);
        LA[index2] += fact * ffy * (A[index2 - yp] + A[index2 - ym] - 2.0 * A[index2]);
        index++;
        index2++;
        index  += h * static_cast<size_t>(d - 1);
        index2 += h * static_cast<size_t>(d - 1);
      }
    } else {
      ym = -h;
      yp =  h;
      fact = factx * factz;
      for (i=0; i<w; i++) {
        if (i == 1) fact *= ifactx;
        if (i == w-1) fact *= factx;
        LA[index]  += fact * ffy * (A[index + yp] - A[index]);
        LA[index2] += fact * ffy * (A[index2 + ym] - A[index2]);
        index++;
        index2++;
        fact *= ifactz;
        for (k=1; k<h-1; k++) {
          // y gradient (+ x, z terms of laplacian, calculated above and below)
          LA[index]  += fact * ffy * (A[index + yp] - A[index]);
          LA[index2] += fact * ffy * (A[index2 + ym] - A[index2]);
          index++;
          index2++;
        }
        fact *= factz;
        LA[index]  += fact * ffy * (A[index + yp] - A[index]);
        LA[index2] += fact * ffy * (A[index2 + ym] - A[index2]);
        index++;
        index2++;
        index  += h * static_cast<size_t>(d - 1);
        index2 += h * static_cast<size_t>(d - 1);
      }
    }

  // Now adding all z components
    // All z components except on z edges
    index = 1; // Skip first element (in bottom slab)
    fact = factx * facty;
    for (i=0; i<w; i++) { // full range of x
      if (i == 1) fact *= ifactx;
      if (i == w-1) fact *= factx;
      for (k=1; k<h-1; k++) {
        LA[index] += fact * ffz * (A[index + zm] + A[index + zp] - 2.0 * A[index]);
        index++;
      }
      fact *= ifacty;
      index += 2; // skip edge slabs
      for (j=1; j<d-1; j++) { // full range of y
        for (k=1; k<h-1; k++) {
          LA[index] += fact * ffz * (A[index + zm] + A[index + zp] - 2.0 * A[index]);
          index++;
        }
        index += 2; // skip edge slabs
      }
      fact *= facty;
      for (k=1; k<h-1; k++) {
        LA[index] += fact * ffz * (A[index + zm] + A[index + zp] - 2.0 * A[index]);
        index++;
      }
      index += 2; // skip edge slabs
    }
    // Edges along z (z components onlz)
    index = 0; // Follows bottom slab
    index2 = h - 1; // Follows top slab
    if (periodic[2]) {
      zm = h - 1;
      zp = 1;
      fact = factx * facty;
      for (i=0; i<w; i++) {
        if (i == 1) fact *= ifactx;
        if (i == w-1) fact *= factx;
        LA[index]  += fact * ffz * (A[index + zm] + A[index + zp] - 2.0 * A[index]);
        LA[index2] += fact * ffz * (A[index2 - zp] + A[index2 - zm] - 2.0 * A[index2]);
        index  += h;
        index2 += h;
        fact *= ifacty;
        for (j=1; j<d-1; j++) {
          LA[index]  += fact * ffz * (A[index + zm] + A[index + zp] - 2.0 * A[index]);
          LA[index2] += fact * ffz * (A[index2 - zp] + A[index2 - zm] - 2.0 * A[index2]);
          index  += h;
          index2 += h;
        }
        fact *= facty;
        LA[index]  += fact * ffz * (A[index + zm] + A[index + zp] - 2.0 * A[index]);
        LA[index2] += fact * ffz * (A[index2 - zp] + A[index2 - zm] - 2.0 * A[index2]);
        index  += h;
        index2 += h;
      }
    } else {
      zm = -1;
      zp = 1;
      fact = factx * facty;
      for (i=0; i<w; i++) {
        if (i == 1) fact *= ifactx;
        if (i == w-1) fact *= factx;
        LA[index]  += fact * ffz * (A[index + zp] - A[index]);
        LA[index2] += fact * ffz * (A[index2 + zm] - A[index2]);
        index  += h;
        index2 += h;
        fact *= ifacty;
        for (j=1; j<d-1; j++) {
          // z gradient (+ x, y terms of laplacian, calculated above)
          LA[index]  += fact * ffz * (A[index + zp] - A[index]);
          LA[index2] += fact * ffz * (A[index2 + zm] - A[index2]);
          index  += h;
          index2 += h;
        }
        fact *= facty;
        LA[index]  += fact * ffz * (A[index + zp] - A[index]);
        LA[index2] += fact * ffz * (A[index2 + zm] - A[index2]);
        index  += h;
        index2 += h;
      }
    }
  }
}


/*
/// Inversion of preconditioner matrix (e.g. diagonal of the Laplacian)
void integrate_potential::asolve(const std::vector<cvm::real> &b, std::vector<cvm::real> &x)
{
  for (size_t i=0; i<int(nt); i++) {
    x[i] = b[i] * inv_lap_diag[i]; // Jacobi preconditioner - little benefit in tests so far
  }
  return;
}*/


// b : RHS of equation
// x : initial guess for the solution; output is solution
// itol : convergence criterion
void integrate_potential::nr_linbcg_sym(const std::vector<cvm::real> &b, std::vector<cvm::real> &x, const cvm::real &tol,
  const int itmax, int &iter, cvm::real &err)
{
  cvm::real ak,akden,bk,bkden,bknum,bnrm;
  const cvm::real EPS=1.0e-14;
  int j;
  std::vector<cvm::real> p(nt), r(nt), z(nt);

  iter=0;
  atimes(x,r);
  for (j=0;j<int(nt);j++) {
    r[j]=b[j]-r[j];
  }
  bnrm=l2norm(b);
  if (bnrm < EPS) {
    return; // Target is zero, will break relative error calc
  }
//   asolve(r,z); // precon
  bkden = 1.0;
  while (iter < itmax) {
    ++iter;
    for (bknum=0.0,j=0;j<int(nt);j++) {
      bknum += r[j]*r[j];  // precon: z[j]*r[j]
    }
    if (iter == 1) {
      for (j=0;j<int(nt);j++) {
        p[j] = r[j];  // precon: p[j] = z[j]
      }
    } else {
      bk=bknum/bkden;
      for (j=0;j<int(nt);j++) {
        p[j] = bk*p[j] + r[j];  // precon:  bk*p[j] + z[j]
      }
    }
    bkden = bknum;
    atimes(p,z);
    for (akden=0.0,j=0;j<int(nt);j++) {
      akden += z[j]*p[j];
    }
    ak = bknum/akden;
    for (j=0;j<int(nt);j++) {
      x[j] += ak*p[j];
      r[j] -= ak*z[j];
    }
//     asolve(r,z);  // precon
    err = l2norm(r)/bnrm;
    if (cvm::debug())
      std::cout << "iter=" << std::setw(4) << iter+1 << std::setw(12) << err << std::endl;
    if (err <= tol)
      break;
  }
}

cvm::real integrate_potential::l2norm(const std::vector<cvm::real> &x)
{
  size_t i;
  cvm::real sum = 0.0;
  for (i=0;i<x.size();i++)
    sum += x[i]*x[i];
  return sqrt(sum);
}
