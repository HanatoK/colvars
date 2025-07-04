#include "colvargrid_integrate.h"

#include <iostream>
#include <iomanip>


colvargrid_integrate::colvargrid_integrate(std::vector<colvar *> &colvars,
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


colvargrid_integrate::colvargrid_integrate(std::shared_ptr<colvar_grid_gradient> gradients)
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


int colvargrid_integrate::integrate(const int itmax, const cvm::real &tol, cvm::real & err, bool verbose)
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


void colvargrid_integrate::set_div()
{
  if (nd == 1) return;
  for (std::vector<int> ix = new_index(); index_ok(ix); incr(ix)) {
    update_div_local(ix);
  }
}


void colvargrid_integrate::update_div_neighbors(const std::vector<int> &ix0)
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


void colvargrid_integrate::get_grad(cvm::real * g, std::vector<int> &ix)
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


void colvargrid_integrate::update_div_local(const std::vector<int> &ix0)
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
void colvargrid_integrate::atimes(const std::vector<cvm::real> &A, std::vector<cvm::real> &LA)
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
void colvargrid_integrate::asolve(const std::vector<cvm::real> &b, std::vector<cvm::real> &x)
{
  for (size_t i=0; i<int(nt); i++) {
    x[i] = b[i] * inv_lap_diag[i]; // Jacobi preconditioner - little benefit in tests so far
  }
  return;
}*/


// b : RHS of equation
// x : initial guess for the solution; output is solution
// itol : convergence criterion
void colvargrid_integrate::nr_linbcg_sym(const std::vector<cvm::real> &b, std::vector<cvm::real> &x, const cvm::real &tol,
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

cvm::real colvargrid_integrate::l2norm(const std::vector<cvm::real> &x)
{
  size_t i;
  cvm::real sum = 0.0;
  for (i=0;i<x.size();i++)
    sum += x[i]*x[i];
  return sqrt(sum);
}
