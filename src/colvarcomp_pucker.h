// -*- c++ -*-
#ifndef COLVARCOMP_PUCKER_H
#define COLVARCOMP_PUCKER_H

#include "colvarcomp.h"

class colvar::cpQ : public colvar::cvc
{
protected:
  /// Atom group
  cvm::atom_group  *atoms = nullptr;
  cvm::real A;
  cvm::real B;
  cvm::real C;
  std::vector<cvm::rvector> dA_dr;
  std::vector<cvm::rvector> dB_dr;
  std::vector<cvm::rvector> dC_dr;
public:
  cpQ();
  virtual int init(std::string const &conf);
  virtual void calc_value();
  virtual void calc_gradients();
};

class colvar::cptheta : public colvar::cvc
{
protected:
  /// Atom group
  cvm::atom_group  *atoms = nullptr;
  cvm::real A;
  cvm::real B;
  cvm::real C;
  std::vector<cvm::rvector> dA_dr;
  std::vector<cvm::rvector> dB_dr;
  std::vector<cvm::rvector> dC_dr;
public:
  cptheta();
  virtual int init(std::string const &conf);
  virtual void calc_value();
  virtual void calc_gradients();
};

class colvar::cpphi : public colvar::cvc
{
protected:
  /// Atom group
  cvm::atom_group  *atoms = nullptr;
  cvm::real A;
  cvm::real B;
  cvm::real C;
  std::vector<cvm::rvector> dA_dr;
  std::vector<cvm::rvector> dB_dr;
  std::vector<cvm::rvector> dC_dr;
public:
  cpphi();
  virtual int init(std::string const &conf);
  virtual void calc_value();
  virtual void calc_gradients();
};

#endif // COLVARCOMP_PUCKER_H

