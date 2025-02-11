// -*- c++ -*-

// This file is part of the Collective Variables module (Colvars).
// The original version of Colvars and its updates are located at:
// https://github.com/Colvars/colvars
// Please update all Colvars source files before making any changes.
// If you wish to distribute your changes, please submit them to the
// Colvars repository at GitHub.

#ifndef COLVARATOMS_H
#define COLVARATOMS_H

#include "colvarmodule.h"
#include "colvarproxy.h"
#include "colvarparse.h"
#include "colvardeps.h"

template <typename T1, typename T2>
struct rotation_derivative;


/// \brief Stores numeric id, mass and all mutable data for an atom,
/// mostly used by a \link colvar::cvc \endlink
///
/// This class may be used to keep atomic data such as id, mass,
/// position and collective variable derivatives) altogether.
/// There may be multiple instances with identical
/// numeric id, all acting independently: forces communicated through
/// these instances will be summed together.

class colvarmodule::atom {

protected:

  /// Index in the colvarproxy arrays (\b NOT in the global topology!)
  int index;

public:

  /// Identifier for the MD program (0-based)
  int             id;

  /// Mass
  cvm::real       mass;

  /// Charge
  cvm::real       charge;

  /// \brief Current position (copied from the program, can be
  /// modified if necessary)
  cvm::atom_pos   pos;

  /// \brief Current velocity (copied from the program, can be
  /// modified if necessary)
  cvm::rvector    vel;

  /// \brief System force at the previous step (copied from the
  /// program, can be modified if necessary)
  cvm::rvector    total_force;

  /// \brief Gradient of a scalar collective variable with respect
  /// to this atom
  ///
  /// This can only handle a scalar collective variable (i.e. when
  /// the \link colvarvalue::real_value \endlink member is used
  /// from the \link colvarvalue \endlink class), which is also the
  /// most frequent case. For more complex types of \link
  /// colvarvalue \endlink objects, atomic gradients should be
  /// defined within the specific \link colvar::cvc \endlink
  /// implementation
  cvm::rvector   grad;

  /// \brief Default constructor (sets index and id both to -1)
  atom();

  /// \brief Initialize an atom for collective variable calculation
  /// and get its internal identifier \param atom_number Atom index in
  /// the system topology (1-based)
  atom(int atom_number);

  /// \brief Initialize an atom for collective variable calculation
  /// and get its internal identifier \param residue Residue number
  /// \param atom_name Name of the atom in the residue \param
  /// segment_id For PSF topologies, the segment identifier; for other
  /// type of topologies, may not be required
  atom(cvm::residue_id const &residue,
       std::string const     &atom_name,
       std::string const     &segment_id);

  /// Copy constructor
  atom(atom const &a);

  /// Destructor
  ~atom();

  /// Assignment operator (added to appease LGTM)
  atom & operator = (atom const &a);

  /// Set mutable data (everything except id and mass) to zero
  inline void reset_data()
  {
    pos = cvm::atom_pos(0.0);
    vel = grad = total_force = cvm::rvector(0.0);
  }

  /// Get the latest value of the mass
  inline void update_mass()
  {
    colvarproxy *p = cvm::proxy;
    mass = p->get_atom_mass(index);
  }

  /// Get the latest value of the charge
  inline void update_charge()
  {
    colvarproxy *p = cvm::proxy;
    charge = p->get_atom_charge(index);
  }

  /// Get the current position
  inline void read_position()
  {
    pos = (cvm::proxy)->get_atom_position(index);
  }

  /// Get the current velocity
  inline void read_velocity()
  {
    vel = (cvm::proxy)->get_atom_velocity(index);
  }

  /// Get the total force
  inline void read_total_force()
  {
    total_force = (cvm::proxy)->get_atom_total_force(index);
  }

  /// \brief Apply a force to the atom
  ///
  /// Note: the force is not applied instantly, but will be used later
  /// by the MD integrator (the colvars module does not integrate
  /// equations of motion.
  ///
  /// Multiple calls to this function by either the same
  /// \link atom \endlink object or different objects with identical
  /// \link id \endlink will all be added together.
  inline void apply_force(cvm::rvector const &new_force) const
  {
    (cvm::proxy)->apply_atom_force(index, new_force);
  }
};



/// \brief Group of \link atom \endlink objects, mostly used by a
/// \link colvar::cvc \endlink object to gather all atomic data
class colvarmodule::atom_group
  : public colvarparse, public colvardeps
{
public:


  /// \brief Default constructor
  atom_group();

  /// \brief Create a group object, assign a name to it
  atom_group(char const *key);

  /// \brief Initialize the group after a (temporary) vector of atoms
  atom_group(std::vector<cvm::atom> const &atoms_in);

  /// \brief Destructor
  ~atom_group() override;

  /// \brief Optional name to reuse properties of this in other groups
  std::string name;

  /// \brief Keyword used to define the group
  // TODO Make this field part of the data structures that link a group to a CVC
  std::string key;

  /// \brief Set default values for common flags
  int init();

  /// \brief Initialize dependency tree
  int init_dependencies() override;

  /// \brief Update data required to calculate cvc's
  int setup();

  /// \brief Initialize the group by looking up its configuration
  /// string in conf and parsing it
  int parse(std::string const &conf);

  int add_atom_numbers(std::string const &numbers_conf);
  int add_atoms_of_group(atom_group const * ag);
  int add_index_group(std::string const &index_group_name, bool silent = false);
  int add_atom_numbers_range(std::string const &range_conf);
  int add_atom_name_residue_range(std::string const &psf_segid,
                                  std::string const &range_conf);
  int parse_fitting_options(std::string const &group_conf);

  /// \brief Add an atom object to this group
  int add_atom(cvm::atom const &a);

  /// \brief Add an atom ID to this group (the actual atomicdata will be not be handled by the group)
  int add_atom_id(int aid);

  /// \brief Remove an atom object from this group
  int remove_atom(cvm::atom_iter ai);

  /// Set this group as a dummy group (no actual atoms)
  int set_dummy();

  /// If this group is dummy, set the corresponding position
  int set_dummy_pos(cvm::atom_pos const &pos);

  /// \brief Print the updated the total mass and charge of a group.
  /// This is needed in case the hosting MD code has an option to
  /// change atom masses after their initialization.
  void print_properties(std::string const &colvar_name, int i, int j);

  /// \brief Implementation of the feature list for atom group
  static std::vector<feature *> ag_features;

  /// \brief Implementation of the feature list accessor for atom group
  const std::vector<feature *> &features() const override { return ag_features; }

  std::vector<feature *> &modify_features() override { return ag_features; }

  static void delete_features()
  {
    for (size_t i = 0; i < ag_features.size(); i++) {
      delete ag_features[i];
    }
    ag_features.clear();
  }

protected:

  /// \brief Array of atom objects
  std::vector<cvm::atom> atoms;

  /// \brief Internal atom IDs for host code
  std::vector<int> atoms_ids;

  /// Sorted list of internal atom IDs (populated on-demand by
  /// create_sorted_ids); used to read coordinate files
  std::vector<int> sorted_atoms_ids;

  /// Map entries of sorted_atoms_ids onto the original positions in the group
  std::vector<int> sorted_atoms_ids_map;

  /// \brief Dummy atom position
  cvm::atom_pos dummy_atom_pos;

  /// \brief Index in the colvarproxy arrays (if the group is scalable)
  int index;

  /// \brief The temporary forces acting on the main group atoms.
  ///        Currently this is only used for calculating the fitting group forces for
  ///        non-scalar components.
  std::vector<cvm::rvector> group_forces;

public:

  /*! @class group_force_object
   *  @brief A helper class for applying forces on an atom group in a way that
   *         is aware of the fitting group. NOTE: you are encouraged to use
   *         get_group_force_object() to get an instance of group_force_object
   *         instead of constructing directly.
   */
  class group_force_object {
  public:
    /*! @brief Constructor of group_force_object
     *  @param ag The pointer to the atom group that forces will be applied on.
     */
    group_force_object(cvm::atom_group* ag);
    /*! @brief Destructor of group_force_object
     */
    ~group_force_object();
    /*! @brief Apply force to atom i
     *  @param i The i-th of atom in the atom group.
     *  @param force The force being added to atom i.
     *
     * The function can be used as follows,
     * @code
     *       // In your colvar::cvc::apply_force() loop of a component:
     *       auto ag_force = atoms->get_group_force_object();
     *       for (ia = 0; ia < atoms->size(); ia++) {
     *         const cvm::rvector f = compute_force_on_atom_ia();
     *         ag_force.add_atom_force(ia, f);
     *       }
     * @endcode
     * There are actually two scenarios under the hood:
     * (i) If the atom group does not have a fitting group, then the force is
     *     added to atom i directly;
     * (ii) If the atom group has a fitting group, the force on atom i will just
     *      be temporary stashed into ag->group_forces. At the end of the loop
     *      of apply_force(), the destructor ~group_force_object() will be called,
     *      which then call apply_force_with_fitting_group(). The forces on the
     *      main group will be rotated back by multiplying ag->group_forces with
     *      the inverse rotation. The forces on the fitting group (if
     *      enableFitGradients is on) will be calculated by calling
     *      calc_fit_forces.
     */
    void add_atom_force(size_t i, const cvm::rvector& force);
  private:
    cvm::atom_group* m_ag;
    cvm::atom_group* m_group_for_fit;
    bool m_has_fitting_force;
    void apply_force_with_fitting_group();
  };

  group_force_object get_group_force_object();

  inline cvm::atom & operator [] (size_t const i)
  {
    return atoms[i];
  }

  inline cvm::atom const & operator [] (size_t const i) const
  {
    return atoms[i];
  }

  inline cvm::atom_iter begin()
  {
    return atoms.begin();
  }

  inline cvm::atom_const_iter begin() const
  {
    return atoms.begin();
  }

  inline cvm::atom_iter end()
  {
    return atoms.end();
  }

  inline cvm::atom_const_iter end() const
  {
    return atoms.end();
  }

  inline size_t size() const
  {
    return atoms.size();
  }

  /// \brief If this option is on, this group merely acts as a wrapper
  /// for a fixed position; any calls to atoms within or to
  /// functions that return disaggregated data will fail
  bool b_dummy;

  /// Internal atom IDs (populated during initialization)
  inline std::vector<int> const &ids() const
  {
    return atoms_ids;
  }

  std::string const print_atom_ids() const;

  /// Allocates and populates sorted_ids and sorted_ids_map
  int create_sorted_ids();

  /// Sorted internal atom IDs (populated on-demand by create_sorted_ids);
  /// used to read coordinate files
  inline std::vector<int> const &sorted_ids() const
  {
    return sorted_atoms_ids;
  }

  /// Map entries of sorted_atoms_ids onto the original positions in the group
  inline std::vector<int> const &sorted_ids_map() const
  {
    return sorted_atoms_ids_map;
  }

  /// Detect whether two groups share atoms
  /// If yes, returns 1-based number of a common atom; else, returns 0
  static int overlap(const atom_group &g1, const atom_group &g2);

  /// The rotation calculated automatically if f_ag_rotate is defined
  cvm::rotation rot;

  /// Rotation derivative;
  rotation_derivative<cvm::atom, cvm::atom_pos>* rot_deriv;

  /// \brief Indicates that the user has explicitly set centerToReference or
  /// rotateReference, and the corresponding reference:
  /// cvc's (eg rmsd, eigenvector) will not override the user's choice
  bool b_user_defined_fit;

  /// \brief use reference coordinates for f_ag_center or f_ag_rotate
  std::vector<cvm::atom_pos> ref_pos;

  /// \brief Center of geometry of the reference coordinates; regardless
  /// of whether f_ag_center is true, ref_pos is centered to zero at
  /// initialization, and ref_pos_cog serves to center the positions
  cvm::atom_pos              ref_pos_cog;

  /// \brief If f_ag_center or f_ag_rotate is true, use this group to
  /// define the transformation (default: this group itself)
  atom_group                *fitting_group;

  /// Total mass of the atom group
  cvm::real total_mass;

  /// Update the total mass of the atom group
  void update_total_mass();

  /// Total charge of the atom group
  cvm::real total_charge;

  /// Update the total mass of the group
  void update_total_charge();

  /// \brief Don't apply any force on this group (use its coordinates
  /// only to calculate a colvar)
  bool noforce;

  /// \brief Get the current positions
  void read_positions();

  /// \brief (Re)calculate the optimal roto-translation
  void calc_apply_roto_translation();

  void setup_rotation_derivative();

  /// \brief Save aside the center of geometry of the reference positions,
  /// then subtract it from them
  ///
  /// In this way it will be possible to use ref_pos also for the
  /// rotational fit.
  /// This is called either by atom_group::parse or by CVCs that assign
  /// reference positions (eg. RMSD, eigenvector).
  void center_ref_pos();

  /// \brief Move all positions
  void apply_translation(cvm::rvector const &t);

  /// \brief Get the current velocities; this must be called always
  /// *after* read_positions(); if f_ag_rotate is defined, the same
  /// rotation applied to the coordinates will be used
  void read_velocities();

  /// \brief Get the current total_forces; this must be called always
  /// *after* read_positions(); if f_ag_rotate is defined, the same
  /// rotation applied to the coordinates will be used
  void read_total_forces();

  /// Call reset_data() for each atom
  inline void reset_atoms_data()
  {
    for (cvm::atom_iter ai = atoms.begin(); ai != atoms.end(); ai++)
      ai->reset_data();
    if (fitting_group)
      fitting_group->reset_atoms_data();
  }

  /// \brief Recompute all mutable quantities that are required to compute CVCs
  int calc_required_properties();

  /// \brief Return a copy of the current atom positions
  std::vector<cvm::atom_pos> positions() const;

  /// \brief Calculate the center of geometry of the atomic positions, assuming
  /// that they are already pbc-wrapped
  int calc_center_of_geometry();

private:

  /// \brief Center of geometry
  cvm::atom_pos cog;

  /// \brief Center of geometry before any fitting
  cvm::atom_pos cog_orig;

  /// \brief Unrotated atom positions for fit gradients
  std::vector<cvm::atom_pos> pos_unrotated;

public:

  /// \brief Return the center of geometry of the atomic positions
  inline cvm::atom_pos center_of_geometry() const
  {
    return cog;
  }

  /// \brief Calculate the center of mass of the atomic positions, assuming that
  /// they are already pbc-wrapped
  int calc_center_of_mass();

private:

  /// \brief Center of mass
  cvm::atom_pos com;

  /// \brief The derivative of a scalar variable with respect to the COM
  // TODO for scalable calculations of more complex variables (e.g. rotation),
  // use a colvarvalue of vectors to hold the entire derivative
  cvm::rvector scalar_com_gradient;

public:

  /// \brief Return the center of mass (COM) of the atomic positions
  inline cvm::atom_pos center_of_mass() const
  {
    return com;
  }

  /// \brief Return previously gradient of scalar variable with respect to the
  /// COM
  inline cvm::rvector center_of_mass_scalar_gradient() const
  {
    return scalar_com_gradient;
  }

  /// \brief Return a copy of the current atom positions, shifted by a constant vector
  std::vector<cvm::atom_pos> positions_shifted(cvm::rvector const &shift) const;

  /// \brief Return a copy of the current atom velocities
  std::vector<cvm::rvector> velocities() const;

  ///\brief Calculate the dipole of the atom group around the specified center
  int calc_dipole(cvm::atom_pos const &dipole_center);

private:

  /// Dipole moment of the atom group
  cvm::rvector dip;

public:

  ///\brief Return the (previously calculated) dipole of the atom group
  inline cvm::rvector dipole() const
  {
    return dip;
  }

  /// \brief Return a copy of the total forces
  std::vector<cvm::rvector> total_forces() const;

  /// \brief Return a copy of the aggregated total force on the group
  cvm::rvector total_force() const;


  /// \brief Shorthand: save the specified gradient on each atom,
  /// weighting with the atom mass (mostly used in combination with
  /// \link center_of_mass() \endlink)
  void set_weighted_gradient(cvm::rvector const &grad);

  /// \brief Calculate the derivatives of the fitting transformation
  void calc_fit_gradients();

/*! @brief  Actual implementation of `calc_fit_gradients` and
 *          `calc_fit_forces`. The template is
 *          used to avoid branching inside the loops in case that the CPU
 *          branch prediction is broken (or further migration to GPU code).
 *  @tparam B_ag_center Centered the reference to origin? This should follow
 *          the value of `is_enabled(f_ag_center)`.
 *  @tparam B_ag_rotate Calculate the optimal rotation? This should follow
 *          the value of `is_enabled(f_ag_rotate)`.
 *  @tparam main_force_accessor_T The type of accessor of the main
 *          group forces or gradients acting on the rotated frame.
 *  @tparam fitting_force_accessor_T The type of accessor of the fitting group
 *          forces or gradients.
 *  @param accessor_main The accessor of the main group forces or gradients.
 *         accessor_main(i) should return the i-th force or gradient of the
 *         rotated main group.
 *  @param accessor_fitting The accessor of the fitting group forces or gradients.
 *         accessor_fitting(j, v) should store/apply the j-th atom gradient or
 *         force in the fitting group.
 *
 *  This function is used to (i) project the gradients of CV with respect to
 *  rotated main group atoms to fitting group atoms, or (ii) project the forces
 *  on rotated main group atoms to fitting group atoms, by the following two steps
 *  (using the goal (ii) for example):
 *  (1) Loop over the positions of main group atoms and call cvm::quaternion::position_derivative_inner
 *      to project the forces on rotated main group atoms to the forces on quaternion.
 *  (2) Loop over the positions of fitting group atoms, compute the gradients of
 *      \f$\mathbf{q}\f$ with respect to the position of each atom, and then multiply
 *      that with the force on \f$\mathbf{q}\f$ (chain rule).
 */
  template <bool B_ag_center, bool B_ag_rotate,
            typename main_force_accessor_T, typename fitting_force_accessor_T>
  void calc_fit_forces_impl(
    main_force_accessor_T accessor_main,
    fitting_force_accessor_T accessor_fitting) const;

/*! @brief  Calculate or apply the fitting group forces from the main group forces.
 *  @tparam main_force_accessor_T The type of accessor of the main
 *          group forces or gradients.
 *  @tparam fitting_force_accessor_T The type of accessor of the fitting group
 *          forces or gradients.
 *  @param accessor_main The accessor of the main group forces or gradients.
 *         accessor_main(i) should return the i-th force or gradient of the
 *         main group.
 *  @param accessor_fitting The accessor of the fitting group forces or gradients.
 *         accessor_fitting(j, v) should store/apply the j-th atom gradient or
 *         force in the fitting group.
 *
 *  This function just dispatches the parameters to calc_fit_forces_impl that really
 *  performs the calculations.
 */
  template <typename main_force_accessor_T, typename fitting_force_accessor_T>
  void calc_fit_forces(
    main_force_accessor_T accessor_main,
    fitting_force_accessor_T accessor_fitting) const;

  /// \brief Derivatives of the fitting transformation
  std::vector<cvm::atom_pos> fit_gradients;

  /// \brief Used by a (scalar) colvar to apply its force on its \link
  /// atom_group \endlink members
  ///
  /// The (scalar) force is multiplied by the colvar gradient for each
  /// atom; this should be used when a colvar with scalar \link
  /// colvarvalue \endlink type is used (this is the most frequent
  /// case: for colvars with a non-scalar type, the most convenient
  /// solution is to sum together the Cartesian forces from all the
  /// colvar components, and use apply_force() or apply_forces()).  If
  /// the group is being rotated to a reference frame (e.g. to express
  /// the colvar independently from the solute rotation), the
  /// gradients are temporarily rotated to the original frame.
  void apply_colvar_force(cvm::real const &force);

  /// \brief Apply a force "to the center of mass", i.e. the force is
  /// distributed on each atom according to its mass
  ///
  /// If the group is being rotated to a reference frame (e.g. to
  /// express the colvar independently from the solute rotation), the
  /// force is rotated back to the original frame.  Colvar gradients
  /// are not used, either because they were not defined (e.g because
  /// the colvar has not a scalar value) or the biases require to
  /// micromanage the force.
  /// This function will be phased out eventually, in favor of
  /// apply_colvar_force() once that is implemented for non-scalar values
  void apply_force(cvm::rvector const &force);

  /// Implements possible actions to be carried out
  /// when a given feature is enabled
  /// This overloads the base function in colvardeps
  void do_feature_side_effects(int id) override;
};


#endif
