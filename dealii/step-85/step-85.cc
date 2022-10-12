/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2021 - 2022 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------
 */

// @sect3{Include files}

// The first include files have all been treated in previous examples.

#include <deal.II/base/function.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>

#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_interface_values.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_tools_cache.h>

#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/q_collection.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/fe_field_function.h>

// The first new header contains some common level set functions.
// For example, the spherical geometry that we use here.
#include <deal.II/base/function_signed_distance.h>

// We also need 3 new headers from the NonMatching namespace.
#include <deal.II/non_matching/fe_immersed_values.h>
#include <deal.II/non_matching/fe_values.h>
#include <deal.II/non_matching/mesh_classifier.h>

#include <fstream>
#include <vector>
#include <algorithm>

// @sect3{The LaplaceSolver class Template}
// We then define the main class that solves the Laplace problem.

namespace Step85
{
  using namespace dealii;

  std::vector<double> load_mat(std::string s)
  {
    std::ifstream is(s);
    std::istream_iterator<double> start(is), end;
    std::vector<double> out(start, end);
    return out;
  }
  template <int dim>
  class RBFinterpolation : public Function<dim>
  {
  public:
    RBFinterpolation(std::string s_controlpx,
                     std::string s_controlpy,
                     std::string s_coeffs,
                     std::string s_shift,
                     std::string s_scale,
                     std::string s_powersx,
                     std::string s_powersy);

    // double value(const Point<dim> &point, const unsigned int component=0) const override
    // {
    //   Vector<double> val_coeffs(n_coeffs);
    //   Vector<double> values(2);

    //   for (unsigned int i = 0; i < n_controlp; i++)
    //   {
    //     val_coeffs[i] = rbf_basis(std::sqrt(std::pow(point[0] - controlPointsx[i], 2)+ std::pow(point[1] - controlPointsy[i], 2)));
    //   }

    //   for (unsigned int i = 0; i < n_powers; i++)
    //   {
    //     val_coeffs[n_controlp+i] = std::pow(point[0], powersx[i]) * std::pow(point[1], powersy[i]);
    //   }

    //   coeffs.Tvmult(values, val_coeffs);
    //   return values[component];
    // }

    void vector_value(const Point<dim> &point, Vector<double> &values) const override
    {
      if (std::abs(std::abs(point[0]) - 1.5) < 1e-4 || std::abs(std::abs(point[1]) - 1.5) < 1e-4)
      {
        values[0] = 0;
        values[1] = 0;
      }
      else
      {
        Vector<double> val_coeffs(n_coeffs);

        for (unsigned int i = 0; i < n_controlp; i++)
        {
          val_coeffs[i] = rbf_basis(std::sqrt(std::pow(point[0] - controlPointsx[i], 2) + std::pow(point[1] - controlPointsy[i], 2)));
        }

        for (unsigned int i = 0; i < n_powers; i++)
        {
          val_coeffs[n_controlp + i] = std::pow(point[0], powersx[i]) * std::pow(point[1], powersy[i]);
        }

        coeffs.Tvmult(values, val_coeffs);
      }
    }

    double rbf_basis(double value) const
    {
      if (std::abs(value) < 1e-8)
      {
        return 0.;
      }
      else
      {
        return std::pow(value, 2) * std::log(value);
      }
    }

  private:
    Vector<double> controlPointsx;
    Vector<double> controlPointsy;

    FullMatrix<double> coeffs;
    Vector<double> powersx;
    Vector<double> powersy;
    Vector<double> shift;
    Vector<double> scale;
    unsigned int n_coeffs;
    unsigned int n_controlp;
    unsigned int n_powers;
  };

  template <int dim>
  RBFinterpolation<dim>::RBFinterpolation(std::string s_controlpx,
                                          std::string s_controlpy,
                                          std::string s_coeffs,
                                          std::string s_shift,
                                          std::string s_scale,
                                          std::string s_powersx,
                                          std::string s_powersy) : RBFinterpolation<dim>::Function(dim)
  {
    std::vector<double> tmp_coeffs = load_mat(s_coeffs);
    coeffs = FullMatrix<double>(tmp_coeffs.size() / dim, dim, &tmp_coeffs[0]);

    std::ifstream is(s_shift);
    std::istream_iterator<double> start(is), end;
    std::vector<double> tmp_vec(start, end);
    shift = Vector<double>(tmp_vec.begin(), tmp_vec.end());

    is = std::ifstream(s_scale);
    start = is;
    tmp_vec = std::vector<double>(start, end);
    scale = Vector<double>(tmp_vec.begin(), tmp_vec.end());

    is = std::ifstream(s_controlpx);
    start = is;
    tmp_vec = std::vector<double>(start, end);
    controlPointsx = Vector<double>(tmp_vec.begin(), tmp_vec.end());

    is = std::ifstream(s_controlpy);
    start = is;
    tmp_vec = std::vector<double>(start, end);
    controlPointsy = Vector<double>(tmp_vec.begin(), tmp_vec.end());

    is = std::ifstream(s_powersx);
    start = is;
    tmp_vec = std::vector<double>(start, end);
    powersx = Vector<double>(tmp_vec.begin(), tmp_vec.end());

    is = std::ifstream(s_powersy);
    start = is;
    tmp_vec = std::vector<double>(start, end);
    powersy = Vector<double>(tmp_vec.begin(), tmp_vec.end());

    n_coeffs = coeffs.m();
    n_controlp = controlPointsx.size();
    n_powers = powersx.size();

    // coeffs.print_formatted(std::cout);
    // shift.print(std::cout);
    // controlPointsx.print(std::cout);
    // controlPointsy.print(std::cout);
    // powersx.print(std::cout);
    // powersy.print(std::cout);
    // scale.print(std::cout);
  }

  template <int dim>
  class SubmeshLevelSet : public Function<dim>
  {
  public:
    /**
     * Constructor, takes the center and radius of the sphereCustom.
     */
    SubmeshLevelSet();

    double
    value(const Point<dim> &point,
          const unsigned int component = 0) const override
    {
      std::vector<Point<dim>> points;
      points.push_back(point);
      auto cell_locations = GridTools::compute_point_locations(*space_grid_tools_cache, points);
      if (std::get<0>(cell_locations).size() > 0)
      {
        return 1;
      }
      else
      {
        return -1;
      }
    }

  private:
    const FullMatrix<double> subm;
    Triangulation<dim> triangulation;
    GridIn<dim> gridin;
    std::unique_ptr<GridTools::Cache<dim, dim>> space_grid_tools_cache;
  };

  template <int dim>
  SubmeshLevelSet<dim>::SubmeshLevelSet()
  {
    // gridin.attach_triangulation(triangulation);
    // std::ifstream f("triangle.msh");
    // gridin.read_msh(f);

    GridGenerator::hyper_ball<dim>(triangulation);
    triangulation.refine_global(2);

    space_grid_tools_cache = std::make_unique<GridTools::Cache<dim, dim>>(triangulation);
  }

  template <int dim>
  class LaplaceSolver
  {
  public:
    LaplaceSolver();

    void run();

  private:
    void make_grid();

    void setup_discrete_level_set();

    void distribute_dofs();

    void initialize_matrices();

    void assemble_system();

    void solve();

    void output_results() const;

    double compute_L2_error() const;

    void pull_back(const Vector<double>& field, Vector<double>& field_deformed);

    bool face_has_ghost_penalty(
        const typename Triangulation<dim>::active_cell_iterator &cell,
        const unsigned int face_index) const;

    const unsigned int fe_degree;

    const Functions::ConstantFunction<dim> rhs_function;
    const Functions::ConstantFunction<dim> boundary_condition;

    Triangulation<dim> triangulation;

    // We need two separate DoFHandlers. The first manages the DoFs for the
    // discrete level set function that describes the geometry of the domain.
    const FE_Q<dim> fe_level_set;
    DoFHandler<dim> level_set_dof_handler;
    Vector<double> level_set;

    // The second DoFHandler manages the DoFs for the solution of the Poisson
    // equation.
    hp::FECollection<dim> fe_collection;
    DoFHandler<dim> dof_handler;
    Vector<double> solution;

    NonMatching::MeshClassifier<dim> mesh_classifier;

    SparsityPattern sparsity_pattern;
    SparseMatrix<double> stiffness_matrix;
    Vector<double> rhs;

    // Optimal transport quantities
    FullMatrix<double> submeshLoaded;
    RBFinterpolation<dim> rbf_transport_map;
    std::unique_ptr<FESystem<dim, dim>> transport_map_fe;
    Vector<double> transport_map;
    std::unique_ptr<MappingQEulerian<dim, Vector<double>, dim>> transport_mapping;
    std::unique_ptr<DoFHandler<dim, dim>> transport_map_dh;
    std::unique_ptr<DoFHandler<dim>> dof_handler_whole_domain;
  };

  template <int dim>
  LaplaceSolver<dim>::LaplaceSolver()
      : fe_degree(1), rhs_function(0), boundary_condition(1.0), fe_level_set(fe_degree), level_set_dof_handler(triangulation), dof_handler(triangulation), mesh_classifier(level_set_dof_handler, level_set),
        rbf_transport_map("./controlpx.txt", "./controlpy.txt", "./coeffs.txt", "./shift.txt", "./scale.txt", "./powersx.txt", "./powersy.txt")
  {
    // std::ifstream is("./submeshLoaded.txt");
    // std::istream_iterator<double> start(is), end;
    // std::vector<double> numbers(start, end);
    // submeshLoaded = FullMatrix<double>(numbers.size() / 3, 3, &numbers[0]);
    // submeshLoaded.print_formatted(std::cout);
  }

  // @sect3{Setting up the Background Mesh}
  // We generate a background mesh with perfectly Cartesian cells. Our domain is
  // a unit disc centered at the origin, so we need to make the background mesh
  // a bit larger than $[-1, 1]^{\text{dim}}$ to completely cover $\Omega$.
  template <int dim>
  void LaplaceSolver<dim>::make_grid()
  {
    std::cout << "Creating background mesh" << std::endl;

    GridGenerator::hyper_cube(triangulation, -1.5, 1.5);
    triangulation.refine_global(3);
  }

  // @sect3{Setting up the Discrete Level Set Function}
  // The discrete level set function is defined on the whole background mesh.
  // Thus, to set up the DoFHandler for the level set function, we distribute
  // DoFs over all elements in $\mathcal{T}_h$. We then set up the discrete
  // level set function by interpolating onto this finite element space.
  template <int dim>
  void LaplaceSolver<dim>::setup_discrete_level_set()
  {
    std::cout << "Setting up discrete level set function" << std::endl;

    level_set_dof_handler.distribute_dofs(fe_level_set);
    level_set.reinit(level_set_dof_handler.n_dofs());

    // const SphereCustom<dim> level_func;
    const SubmeshLevelSet<dim> level_func;

    VectorTools::interpolate(level_set_dof_handler,
                             level_func,
                             level_set);
  }

  // @sect3{Setting up the Finite Element Space}
  // To set up the finite element space $V_\Omega^h$, we will use 2 different
  // elements: FE_Q and FE_Nothing. For better readability we define an enum for
  // the indices in the order we store them in the hp::FECollection.
  enum ActiveFEIndex
  {
    lagrange = 0,
    nothing = 1
  };

  // We then use the MeshClassifier to check LocationToLevelSet for each cell in
  // the mesh and tell the DoFHandler to use FE_Q on elements that are inside or
  // intersected, and FE_Nothing on the elements that are outside.
  template <int dim>
  void LaplaceSolver<dim>::distribute_dofs()
  {
    std::cout << "Distributing degrees of freedom" << std::endl;

    fe_collection.push_back(FE_Q<dim>(fe_degree));
    fe_collection.push_back(FE_Nothing<dim>());

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      const NonMatching::LocationToLevelSet cell_location =
          mesh_classifier.location_to_level_set(cell);

      if (cell_location == NonMatching::LocationToLevelSet::outside)
        cell->set_active_fe_index(ActiveFEIndex::nothing);
      else
        cell->set_active_fe_index(ActiveFEIndex::lagrange);
    }

    dof_handler.distribute_dofs(fe_collection);
  }

  // @sect3{Sparsity Pattern}
  // The added ghost penalty results in a sparsity pattern similar to a DG
  // method with a symmetric-interior-penalty term. Thus, we can use the
  // make_flux_sparsity_pattern() function to create it. However, since the
  // ghost-penalty terms only act on the faces in $\mathcal{F}_h$, we can pass
  // in a lambda function that tells make_flux_sparsity_pattern() over which
  // faces the flux-terms appear. This gives us a sparsity pattern with minimal
  // number of entries. When passing a lambda function,
  // make_flux_sparsity_pattern requires us to also pass cell and face coupling
  // tables to it. If the problem was vector-valued, these tables would allow us
  // to couple only some of the vector components. This is discussed in step-46.
  template <int dim>
  void LaplaceSolver<dim>::initialize_matrices()
  {
    std::cout << "Initializing matrices" << std::endl;

    const auto face_has_flux_coupling = [&](const auto &cell,
                                            const unsigned int face_index)
    {
      return this->face_has_ghost_penalty(cell, face_index);
    };

    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());

    const unsigned int n_components = fe_collection.n_components();
    Table<2, DoFTools::Coupling> cell_coupling(n_components, n_components);
    Table<2, DoFTools::Coupling> face_coupling(n_components, n_components);
    cell_coupling[0][0] = DoFTools::always;
    face_coupling[0][0] = DoFTools::always;

    const AffineConstraints<double> constraints;
    const bool keep_constrained_dofs = true;

    DoFTools::make_flux_sparsity_pattern(dof_handler,
                                         dsp,
                                         constraints,
                                         keep_constrained_dofs,
                                         cell_coupling,
                                         face_coupling,
                                         numbers::invalid_subdomain_id,
                                         face_has_flux_coupling);
    sparsity_pattern.copy_from(dsp);

    stiffness_matrix.reinit(sparsity_pattern);
    solution.reinit(dof_handler.n_dofs());
    rhs.reinit(dof_handler.n_dofs());
  }

  // The following function describes which faces are part of the set
  // $\mathcal{F}_h$. That is, it returns true if the face of the incoming cell
  // belongs to the set $\mathcal{F}_h$.
  template <int dim>
  bool LaplaceSolver<dim>::face_has_ghost_penalty(
      const typename Triangulation<dim>::active_cell_iterator &cell,
      const unsigned int face_index) const
  {
    if (cell->at_boundary(face_index))
      return false;

    const NonMatching::LocationToLevelSet cell_location =
        mesh_classifier.location_to_level_set(cell);

    const NonMatching::LocationToLevelSet neighbor_location =
        mesh_classifier.location_to_level_set(cell->neighbor(face_index));

    if (cell_location == NonMatching::LocationToLevelSet::intersected &&
        neighbor_location != NonMatching::LocationToLevelSet::outside)
      return true;

    if (neighbor_location == NonMatching::LocationToLevelSet::intersected &&
        cell_location != NonMatching::LocationToLevelSet::outside)
      return true;

    return false;
  }

  // @sect3{Assembling the System}
  template <int dim>
  void LaplaceSolver<dim>::assemble_system()
  {
    std::cout << "Assembling" << std::endl;

    const unsigned int n_dofs_per_cell = fe_collection[0].dofs_per_cell;
    FullMatrix<double> local_stiffness(n_dofs_per_cell, n_dofs_per_cell);
    Vector<double> local_rhs(n_dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);

    const double ghost_parameter = 0.5;
    const double nitsche_parameter = 5 * (fe_degree + 1) * fe_degree;

    // Since the ghost penalty is similar to a DG flux term, the simplest way to
    // assemble it is to use an FEInterfaceValues object.
    const QGauss<dim - 1> face_quadrature(fe_degree + 1);
    FEInterfaceValues<dim> fe_interface_values(fe_collection[0],
                                               face_quadrature,
                                               update_gradients |
                                                   update_JxW_values |
                                                   update_normal_vectors);

    // As we iterate over the cells in the mesh, we would in principle have to
    // do the following on each cell, $T$,
    //
    // 1. Construct one quadrature rule to integrate over the intersection with
    // the domain, $T \cap \Omega$, and one quadrature rule to integrate over
    // the intersection with the boundary, $T \cap \Gamma$.
    // 2. Create FEValues-like objects with the new quadratures.
    // 3. Assemble the local matrix using the created FEValues-objects.
    //
    // To make the assembly easier, we use the class NonMatching::FEValues,
    // which does the above steps 1 and 2 for us. The algorithm @cite saye_2015
    // that is used to generate the quadrature rules on the intersected cells
    // uses a 1-dimensional quadrature rule as base. Thus, we pass a 1D
    // Gauss--Legendre quadrature to the constructor of NonMatching::FEValues.
    // On the non-intersected cells, a tensor product of this 1D-quadrature will
    // be used.
    //
    // As stated in the introduction, each cell has 3 different regions: inside,
    // surface, and outside, where the level set function in each region is
    // negative, zero, and positive. We need an UpdateFlags variable for each
    // such region. These are stored on an object of type
    // NonMatching::RegionUpdateFlags, which we pass to NonMatching::FEValues.
    const QGauss<1> quadrature_1D(fe_degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside = update_values | update_gradients |
                                 update_JxW_values | update_quadrature_points;
    region_update_flags.surface = update_values | update_gradients |
                                  update_JxW_values | update_quadrature_points |
                                  update_normal_vectors;

    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);

    // As we iterate over the cells, we don't need to do anything on the cells
    // that have FENothing elements. To disregard them we use an iterator
    // filter.
    for (const auto &cell :
         dof_handler.active_cell_iterators() |
             IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::lagrange))
    {
      local_stiffness = 0;
      local_rhs = 0;

      const double cell_side_length = cell->minimum_vertex_distance();

      // First, we call the reinit function of our NonMatching::FEValues
      // object. In the background, NonMatching::FEValues uses the
      // MeshClassifier passed to its constructor to check if the incoming
      // cell is intersected. If that is the case, NonMatching::FEValues calls
      // the NonMatching::QuadratureGenerator in the background to create the
      // immersed quadrature rules.
      non_matching_fe_values.reinit(cell);

      // After calling reinit, we can retrieve a dealii::FEValues object with
      // quadrature points that corresponds to integrating over the inside
      // region of the cell. This is the object we use to do the local
      // assembly. This is similar to how hp::FEValues builds dealii::FEValues
      // objects. However, one difference here is that the dealii::FEValues
      // object is returned as an optional. This is a type that wraps an
      // object that may or may not be present. This requires us to add an
      // if-statement to check if the returned optional contains a value,
      // before we use it. This might seem odd at first. Why does the function
      // not just return a reference to a const FEValues<dim>? The reason is
      // that in an immersed method, we have essentially no control of how the
      // cuts occur. Even if the cell is formally intersected: $T \cap \Omega
      // \neq \emptyset$, it might be that the cut is only of floating point
      // size $|T \cap \Omega| \sim \epsilon$. When this is the case, we can
      // not expect that the algorithm that generates the quadrature rule
      // produces anything useful. It can happen that the algorithm produces 0
      // quadrature points. When this happens, the returned optional will not
      // contain a value, even if the cell is formally intersected.
      const std_cxx17::optional<FEValues<dim>> &inside_fe_values =
          non_matching_fe_values.get_inside_fe_values();

      if (inside_fe_values)
        for (const unsigned int q :
             inside_fe_values->quadrature_point_indices())
        {
          const Point<dim> &point = inside_fe_values->quadrature_point(q);
          for (const unsigned int i : inside_fe_values->dof_indices())
          {
            for (const unsigned int j : inside_fe_values->dof_indices())
            {
              local_stiffness(i, j) +=
                  inside_fe_values->shape_grad(i, q) *
                  inside_fe_values->shape_grad(j, q) *
                  inside_fe_values->JxW(q);
            }
            local_rhs(i) += rhs_function.value(point) *
                            inside_fe_values->shape_value(i, q) *
                            inside_fe_values->JxW(q);
          }
        }

      // In the same way, we can use NonMatching::FEValues to retrieve an
      // FEFaceValues-like object to integrate over $T \cap \Gamma$. The only
      // thing that is new here is the type of the object. The transformation
      // from quadrature weights to JxW-values is different for surfaces, so
      // we need a new class: NonMatching::FEImmersedSurfaceValues. In
      // addition to the ordinary functions shape_value(..), shape_grad(..),
      // etc., one can use its normal_vector(..)-function to get an outward
      // normal to the immersed surface, $\Gamma$. In terms of the level set
      // function, this normal reads
      // @f{equation*}
      //   n = \frac{\nabla \psi}{\| \nabla \psi \|}.
      // @f}
      // An additional benefit of std::optional is that we do not need any
      // other check for whether we are on intersected cells: In case we are
      // on an inside cell, we get an empty object here.
      const std_cxx17::optional<NonMatching::FEImmersedSurfaceValues<dim>>
          &surface_fe_values = non_matching_fe_values.get_surface_fe_values();

      if (surface_fe_values)
      {
        for (const unsigned int q :
             surface_fe_values->quadrature_point_indices())
        {
          const Point<dim> &point =
              surface_fe_values->quadrature_point(q);
          const Tensor<1, dim> &normal =
              surface_fe_values->normal_vector(q);
          for (const unsigned int i : surface_fe_values->dof_indices())
          {
            for (const unsigned int j :
                 surface_fe_values->dof_indices())
            {
              local_stiffness(i, j) +=
                  (-normal * surface_fe_values->shape_grad(i, q) *
                       surface_fe_values->shape_value(j, q) +
                   -normal * surface_fe_values->shape_grad(j, q) *
                       surface_fe_values->shape_value(i, q) +
                   nitsche_parameter / cell_side_length *
                       surface_fe_values->shape_value(i, q) *
                       surface_fe_values->shape_value(j, q)) *
                  surface_fe_values->JxW(q);
            }
            local_rhs(i) +=
                boundary_condition.value(point) *
                (nitsche_parameter / cell_side_length *
                     surface_fe_values->shape_value(i, q) -
                 normal * surface_fe_values->shape_grad(i, q)) *
                surface_fe_values->JxW(q);
          }
        }
      }

      cell->get_dof_indices(local_dof_indices);

      stiffness_matrix.add(local_dof_indices, local_stiffness);
      rhs.add(local_dof_indices, local_rhs);

      // The assembly of the ghost penalty term is straight forward. As we
      // iterate over the local faces, we first check if the current face
      // belongs to the set $\mathcal{F}_h$. The actual assembly is simple
      // using FEInterfaceValues. Assembling in this we will traverse each
      // internal face in the mesh twice, so in order to get the penalty
      // constant we expect, we multiply the penalty term with a factor 1/2.
      for (unsigned int f : cell->face_indices())
        if (face_has_ghost_penalty(cell, f))
        {
          const unsigned int invalid_subface =
              numbers::invalid_unsigned_int;

          fe_interface_values.reinit(cell,
                                     f,
                                     invalid_subface,
                                     cell->neighbor(f),
                                     cell->neighbor_of_neighbor(f),
                                     invalid_subface);

          const unsigned int n_interface_dofs =
              fe_interface_values.n_current_interface_dofs();
          FullMatrix<double> local_stabilization(n_interface_dofs,
                                                 n_interface_dofs);
          for (unsigned int q = 0;
               q < fe_interface_values.n_quadrature_points;
               ++q)
          {
            const Tensor<1, dim> normal = fe_interface_values.normal(q);
            for (unsigned int i = 0; i < n_interface_dofs; ++i)
              for (unsigned int j = 0; j < n_interface_dofs; ++j)
              {
                local_stabilization(i, j) +=
                    .5 * ghost_parameter * cell_side_length * normal *
                    fe_interface_values.jump_in_shape_gradients(i, q) *
                    normal *
                    fe_interface_values.jump_in_shape_gradients(j, q) *
                    fe_interface_values.JxW(q);
              }
          }

          const std::vector<types::global_dof_index>
              local_interface_dof_indices =
                  fe_interface_values.get_interface_dof_indices();

          stiffness_matrix.add(local_interface_dof_indices,
                               local_stabilization);
        }
    }

    std::map<types::global_dof_index, double> boundary_values;
    VectorTools::interpolate_boundary_values(dof_handler,
                                             0,
                                             Functions::ZeroFunction<2>(),
                                             boundary_values);
    MatrixTools::apply_boundary_values(boundary_values,
                                       stiffness_matrix,
                                       solution,
                                       rhs);
  }

  // @sect3{Solving the System}
  template <int dim>
  void LaplaceSolver<dim>::solve()
  {
    std::cout << "Solving system" << std::endl;

    const unsigned int max_iterations = 2 * solution.size();
    SolverControl solver_control(max_iterations);
    SolverCG<> solver(solver_control);
    solver.solve(stiffness_matrix, solution, rhs, PreconditionIdentity());
  }

  // @sect3{Data Output}
  // Since both DoFHandler instances use the same triangulation, we can add both
  // the level set function and the solution to the same vtu-file. Further, we
  // do not want to output the cells that have LocationToLevelSet value outside.
  // To disregard them, we write a small lambda function and use the
  // set_cell_selection function of the DataOut class.
  template <int dim>
  void LaplaceSolver<dim>::output_results() const
  {
    std::cout << "Writing vtu file" << std::endl;

    DataOut<dim> data_out;
    data_out.add_data_vector(dof_handler, solution, "solution");
    data_out.add_data_vector(level_set_dof_handler, level_set, "level_set");

    // data_out.set_cell_selection(
    //     [this](const typename Triangulation<dim>::cell_iterator &cell)
    //     {
    //       return cell->is_active() &&
    //              mesh_classifier.location_to_level_set(cell) !=
    //                  NonMatching::LocationToLevelSet::outside;
    //     });

    data_out.build_patches();
    std::ofstream output("step-85.vtu");
    data_out.write_vtu(output);
  }

  // @sect3{$L^2$-Error}
  // To test that the implementation works as expected, we want to compute the
  // error in the solution in the $L^2$-norm. The analytical solution to the
  // Poisson problem stated in the introduction reads
  // @f{align*}
  //  u(x) = 1 - \frac{2}{\text{dim}}(\| x \|^2 - 1) , \qquad x \in
  //  \overline{\Omega}.
  // @f}
  // We first create a function corresponding to the analytical solution:
  template <int dim>
  class AnalyticalSolution : public Function<dim>
  {
  public:
    double value(const Point<dim> &point,
                 const unsigned int component = 0) const override;
  };

  template <int dim>
  double AnalyticalSolution<dim>::value(const Point<dim> &point,
                                        const unsigned int component) const
  {
    AssertIndexRange(component, this->n_components);
    (void)component;

    return 1. - 2. / dim * (point.norm_square() - 1.);
  }

  // Of course, the analytical solution, and thus also the error, is only
  // defined in $\overline{\Omega}$. Thus, to compute the $L^2$-error we must
  // proceed in the same way as when we assembled the linear system. We first
  // create an NonMatching::FEValues object.
  template <int dim>
  double LaplaceSolver<dim>::compute_L2_error() const
  {
    std::cout << "Computing L2 error" << std::endl;

    const QGauss<1> quadrature_1D(fe_degree + 1);

    NonMatching::RegionUpdateFlags region_update_flags;
    region_update_flags.inside =
        update_values | update_JxW_values | update_quadrature_points;

    NonMatching::FEValues<dim> non_matching_fe_values(fe_collection,
                                                      quadrature_1D,
                                                      region_update_flags,
                                                      mesh_classifier,
                                                      level_set_dof_handler,
                                                      level_set);

    // We then iterate iterate over the cells that have LocationToLevelSetValue
    // value inside or intersected again. For each quadrature point, we compute
    // the pointwise error and use this to compute the integral.
    const AnalyticalSolution<dim> analytical_solution;
    double error_L2_squared = 0;

    for (const auto &cell :
         dof_handler.active_cell_iterators() |
             IteratorFilters::ActiveFEIndexEqualTo(ActiveFEIndex::lagrange))
    {
      non_matching_fe_values.reinit(cell);

      const std_cxx17::optional<FEValues<dim>> &fe_values =
          non_matching_fe_values.get_inside_fe_values();

      if (fe_values)
      {
        std::vector<double> solution_values(fe_values->n_quadrature_points);
        fe_values->get_function_values(solution, solution_values);

        for (const unsigned int q : fe_values->quadrature_point_indices())
        {
          const Point<dim> &point = fe_values->quadrature_point(q);
          const double error_at_point =
              solution_values.at(q) - analytical_solution.value(point);
          error_L2_squared +=
              std::pow(error_at_point, 2) * fe_values->JxW(q);
        }
      }
    }

    return std::sqrt(error_L2_squared);
  }

  template <int dim>
  void LaplaceSolver<dim>::pull_back(const Vector<double>& field, Vector<double>& field_deformed)
  {
    transport_map_fe = std::make_unique<FESystem<dim, dim>>(
        FE_Q<dim, dim>(fe_degree),
        dim);

    transport_map_dh =
        std::make_unique<DoFHandler<dim, dim>>(triangulation);

    transport_map_dh->distribute_dofs(*transport_map_fe);
    transport_map.reinit(transport_map_dh->n_dofs());

    std::cout << "debug intepolate transport map\n";
    VectorTools::interpolate(*transport_map_dh,
                             rbf_transport_map,
                             transport_map);

    DataOut<dim> data_out;
    std::vector<std::string> solution_names(dim, "phi");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_out.attach_dof_handler(*transport_map_dh);
    data_out.add_data_vector(transport_map, solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    data_out.build_patches();
    std::ofstream output("transport_map.vtu");
    data_out.write_vtu(output);

    std::cout << "debug create eulerian mapping\n";
    transport_mapping = std::make_unique<MappingQEulerian<dim, Vector<double>, dim>>(fe_degree,
        *transport_map_dh,
        transport_map);

    Functions::FEFieldFunction<dim> field_function_(dof_handler, field);

    std::cout << "fe_collection dofs: " << dof_handler.n_dofs() << std::endl;
    dof_handler_whole_domain = std::make_unique<DoFHandler<dim>>(triangulation);
    dof_handler_whole_domain->distribute_dofs(FE_Q<dim>(fe_degree));
    std::cout << "fe whole domain dofs: " << dof_handler_whole_domain->n_dofs() << std::endl;

    Vector<double> field_(dof_handler_whole_domain->n_dofs());
    FETools::interpolate(dof_handler, field, *dof_handler_whole_domain, field_);
    Functions::FEFieldFunction<dim> field_function(*dof_handler_whole_domain, field_);

    // Point<dim> p(0, 0);
    // Vector<double> v(1);
    // field_function.vector_value(p, v);
    // std::cout << "debug vector: ";
    // v.print(std::cout);

    field_deformed.reinit(dof_handler_whole_domain->n_dofs());

    std::cout << "debug interpolate deformed: " << dof_handler_whole_domain->n_dofs() << std::endl;
    VectorTools::interpolate(*transport_mapping,
                             *dof_handler_whole_domain,
                             field_function,
                             field_deformed);
    
    DataOut<dim> data_out_;
    data_out_.add_data_vector(*dof_handler_whole_domain, field_deformed, "deformed");
    data_out_.build_patches();
    std::ofstream output_("deformed.vtu");
    data_out_.write_vtu(output_);
  }

  // @sect3{A Convergence Study}
  // Finally, we do a convergence study to check that the $L^2$-error decreases
  // with the expected rate. We refine the background mesh a few times. In each
  // refinement cycle, we solve the problem, compute the error, and add the
  // $L^2$-error and the mesh size to a ConvergenceTable.
  template <int dim>
  void LaplaceSolver<dim>::run()
  {
    ConvergenceTable convergence_table;
    const unsigned int n_refinements = 3;

    make_grid();
    for (unsigned int cycle = 3; cycle <= n_refinements; cycle++)
    {
      std::cout << "Refinement cycle " << cycle << std::endl;
      triangulation.refine_global(1);
      setup_discrete_level_set();
      std::cout << "Classifying cells" << std::endl;
      mesh_classifier.reclassify();
      distribute_dofs();
      initialize_matrices();
      assemble_system();
      solve();
      output_results();
      Vector<double> deformed;
      pull_back(solution, deformed);

      // const double error_L2 = compute_L2_error();
      // const double cell_side_length =
      //     triangulation.begin_active()->minimum_vertex_distance();

      // convergence_table.add_value("Cycle", cycle);
      // convergence_table.add_value("Mesh size", cell_side_length);
      // convergence_table.add_value("L2-Error", error_L2);

      // convergence_table.evaluate_convergence_rates(
      //     "L2-Error", ConvergenceTable::reduction_rate_log2);
      // convergence_table.set_scientific("L2-Error", true);

      // std::cout << std::endl;
      // convergence_table.write_text(std::cout);
      // std::cout << std::endl;
    }
  }

} // namespace Step85

// @sect3{The main() function}
int main()
{
  const int dim = 2;

  Step85::LaplaceSolver<dim> laplace_solver;
  laplace_solver.run();
}
