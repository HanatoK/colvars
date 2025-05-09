// -*- c++ -*-

// This file is part of the Collective Variables module (Colvars).
// The original version of Colvars and its updates are located at:
// https://github.com/Colvars/colvars
// Please update all Colvars source files before making any changes.
// If you wish to distribute your changes, please submit them to the
// Colvars repository at GitHub.

#include <iostream>

#include "colvarmodule.h"
#include "colvarproxy.h"
#include "colvar.h"
#include "colvarbias_histogram.h"
#include "colvars_memstream.h"


colvarbias_histogram::colvarbias_histogram(char const *key)
  : colvarbias(key),
    grid(NULL), out_name("")
{
  provide(f_cvb_bypass_ext_lagrangian); // Allow histograms of actual cv for extended-Lagrangian
}


int colvarbias_histogram::init(std::string const &conf)
{
  int err = colvarbias::init(conf);
  if (err != COLVARS_OK) {
    return err;
  }
  cvm::main()->cite_feature("Histogram colvar bias implementation");

  enable(f_cvb_scalar_variables);
  enable(f_cvb_history_dependent);

  size_t i;

  get_keyval(conf, "outputFile", out_name, "");
  // Write DX file by default only in dimension >= 3
  std::string default_name_dx = this->num_variables() > 2 ? "" : "none";
  get_keyval(conf, "outputFileDX", out_name_dx, default_name_dx);

  /// with VMD, this may not be an error
  // if ( output_freq == 0 ) {
  //   cvm::error("User required histogram with zero output frequency");
  // }

  colvar_array_size = 0;
  {
    bool colvar_array = false;
    get_keyval(conf, "gatherVectorColvars", colvar_array, colvar_array);

    if (colvar_array) {
      for (i = 0; i < num_variables(); i++) { // should be all vector
        if (colvars[i]->value().type() != colvarvalue::type_vector) {
          cvm::error("Error: used gatherVectorColvars with non-vector colvar.\n", COLVARS_INPUT_ERROR);
          return COLVARS_INPUT_ERROR;
        }
        if (i == 0) {
          colvar_array_size = colvars[i]->value().size();
          if (colvar_array_size < 1) {
            cvm::error("Error: vector variable has dimension less than one.\n", COLVARS_INPUT_ERROR);
            return COLVARS_INPUT_ERROR;
          }
        } else {
          if (colvar_array_size != colvars[i]->value().size()) {
            cvm::error("Error: trying to combine vector colvars of different lengths.\n", COLVARS_INPUT_ERROR);
            return COLVARS_INPUT_ERROR;
          }
        }
      }
    } else {
      for (i = 0; i < num_variables(); i++) { // should be all scalar
        if (colvars[i]->value().type() != colvarvalue::type_scalar) {
          cvm::error("Error: only scalar colvars are supported when gatherVectorColvars is off.\n", COLVARS_INPUT_ERROR);
          return COLVARS_INPUT_ERROR;
        }
      }
    }
  }

  if (colvar_array_size > 0) {
    weights.assign(colvar_array_size, 1.0);
    get_keyval(conf, "weights", weights, weights);
  }

  for (i = 0; i < num_variables(); i++) {
    colvars[i]->enable(f_cv_grid); // Could be a child dependency of a f_cvb_use_grids feature
  }

  grid = new colvar_grid_scalar();
  grid->init_from_colvars(colvars);

  if (is_enabled(f_cvb_bypass_ext_lagrangian)) {
    grid->request_actual_value();
  }

  {
    if (key_lookup(conf, "histogramGrid", &grid_conf) ||
        key_lookup(conf, "grid", &grid_conf)) {
      grid->parse_params(grid_conf);
      grid->check_keywords(grid_conf, "grid");
    }
  }

  return COLVARS_OK;
}


colvarbias_histogram::~colvarbias_histogram()
{
  if (grid) {
    delete grid;
    grid = NULL;
  }
}


int colvarbias_histogram::update()
{
  int error_code = COLVARS_OK;
  // update base class
  error_code |= colvarbias::update();

  if (cvm::debug()) {
    cvm::log("Updating histogram bias " + this->name);
  }

  // assign a valid bin size
  bin.assign(num_variables(), 0);

  if (colvar_array_size == 0) {
    // update indices for scalar values
    size_t i;
    for (i = 0; i < num_variables(); i++) {
      bin[i] = grid->current_bin_scalar(i);
    }

    if (can_accumulate_data()) {
      if (grid->index_ok(bin)) {
        grid->acc_value(bin, 1.0);
      }
    }
  } else {
    // update indices for vector/array values
    size_t iv, i;
    for (iv = 0; iv < colvar_array_size; iv++) {
      for (i = 0; i < num_variables(); i++) {
        bin[i] = grid->current_bin_scalar(i, iv);
      }

      if (grid->index_ok(bin)) {
        grid->acc_value(bin, weights[iv]);
      }
    }
  }

  error_code |= cvm::get_error();
  return error_code;
}


int colvarbias_histogram::write_output_files()
{
  if (!has_data) {
    // nothing to write
    return COLVARS_OK;
  }

  int error_code = COLVARS_OK;

  // Set default filenames, if none have been provided
  if (!cvm::output_prefix().empty()) {
    if (out_name.empty()) {
      out_name = cvm::output_prefix() + "." + this->name + ".dat";
    }
    if (out_name_dx.empty()) {
      out_name_dx = cvm::output_prefix() + "." + this->name + ".dx";
    }
  }

  if (out_name.size() && out_name != "none") {
    cvm::log("Writing the histogram file \""+out_name+"\".\n");
    error_code |= grid->write_multicol(out_name, "histogram output file");
  }

  if (out_name_dx.size() && out_name_dx != "none") {
    cvm::log("Writing the histogram file \""+out_name_dx+"\".\n");
    error_code |= grid->write_opendx(out_name_dx, "histogram DX output file");
  }

  return error_code;
}


std::istream & colvarbias_histogram::read_state_data(std::istream& is)
{
  if (read_state_data_key(is, "grid")) {
    grid->read_raw(is);
  }
  return is;
}


cvm::memory_stream & colvarbias_histogram::read_state_data(cvm::memory_stream& is)
{
  if (read_state_data_key(is, "grid")) {
    grid->read_raw(is);
  }
  return is;
}


std::ostream & colvarbias_histogram::write_state_data(std::ostream& os)
{
  std::ios::fmtflags flags(os.flags());
  os.setf(std::ios::fmtflags(0), std::ios::floatfield);
  write_state_data_key(os, "grid");
  grid->write_raw(os, 8);
  os.flags(flags);
  return os;
}


cvm::memory_stream & colvarbias_histogram::write_state_data(cvm::memory_stream& os)
{
  write_state_data_key(os, "grid");
  grid->write_raw(os);
  return os;
}
