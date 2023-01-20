/*---------------------------------------------------------------------------*\
     ██╗████████╗██╗  ██╗ █████╗  ██████╗ █████╗       ███████╗██╗   ██╗
     ██║╚══██╔══╝██║  ██║██╔══██╗██╔════╝██╔══██╗      ██╔════╝██║   ██║
     ██║   ██║   ███████║███████║██║     ███████║█████╗█████╗  ██║   ██║
     ██║   ██║   ██╔══██║██╔══██║██║     ██╔══██║╚════╝██╔══╝  ╚██╗ ██╔╝
     ██║   ██║   ██║  ██║██║  ██║╚██████╗██║  ██║      ██║      ╚████╔╝
     ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝      ╚═╝       ╚═══╝

 * In real Time Highly Advanced Computational Applications for Finite Volumes
 * Copyright (C) 2017 by the ITHACA-FV authors
-------------------------------------------------------------------------------
License
    This file is part of ITHACA-FV
    ITHACA-FV is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    ITHACA-FV is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU Lesser General Public License for more details.
    You should have received a copy of the GNU Lesser General Public License
    along with ITHACA-FV. If not, see <http://www.gnu.org/licenses/>.
Description
    Example of a heat transfer Reduction Problem
SourceFiles
    02thermalBlock.C
\*---------------------------------------------------------------------------*/

#include <iostream>
#include "fvCFD.H"
#include "IOmanip.H"
#include "Time.H"
#include "laplacianProblem.H"
#include "ReducedLaplacian.H"
#include "ITHACAPOD.H"
#include "ITHACAutilities.H"
#include "simpleControl.H"
#include "fvOptions.H"

#include <Eigen/Dense>
#define _USE_MATH_DEFINES
#include <cmath>

/// \brief Class where the tutorial number 2 is implemented.
/// \details It is a child of the laplacianProblem class and some of its
/// functions are overridden to be adapted to the specific case.
class bunny: public laplacianProblem
{
    public:
        /// Temperature field
        volScalarField& T;
        /// Diffusivity field
        volScalarField& nu;
        /// Source term field
        volScalarField& S;

        explicit bunny(int argc, char* argv[])
            :
            laplacianProblem(argc, argv),
            T(_T()),
            nu(_nu()),
            S(_S())
        {}

        /// It perform an offline Solve
        void offlineSolve(word folder = "./ITHACAoutput/Offline/")
        {
            auto _simple = autoPtr<simpleControl>
                (
                    new simpleControl
                    (
                        _mesh()
                    )
                );

            auto _fvOptions = autoPtr<fv::options>(new fv::options(_mesh()));

            auto _transportProperties = autoPtr<IOdictionary>
                    (
                        new IOdictionary
                        (
                            IOobject
                            (
                                "transportProperties",
                                _runTime().constant(),
                                _mesh(),
                                IOobject::MUST_READ_IF_MODIFIED,
                                IOobject::NO_WRITE
                            )
                        )
                    );

            auto _DT = autoPtr<dimensionedScalar>
                (
                    new dimensionedScalar
                    (
                        _transportProperties().lookup("DT")
                    )
                );

            if (offline)
            {
                ITHACAstream::read_fields(Tfield, "T", folder);
                mu_samples =
                    ITHACAstream::readMatrix(folder + "/mu_samples_mat.txt");
            }
            else
            {
                simpleControl& simple = _simple();
                fv::options& fvOptions = _fvOptions();
                volScalarField& T = _T();
                volScalarField& S = _S();
                dimensionedScalar& DT = _DT();

                Info << " " << T.size() << " " << DT.value() << endl;

                // while (simple.loop())
                // {
                    Info<< "Time = " << _runTime().timeName() << nl << endl;

                    while (simple.correctNonOrthogonal())
                    {
                        fvScalarMatrix TEqn
                        (
                            fvm::ddt(T) - fvm::laplacian(DT, T)
                        ==
                            // S +
                            fvOptions(T)
                        );

                        fvOptions.constrain(TEqn);
                        TEqn.solve();
                        fvOptions.correct(T);
                    }

                    _runTime().write();
                // }

                ITHACAstream::exportSolution(T, name(counter), folder);
                Tfield.append(T.clone());
                counter++;
            }
        }

        /// Define the source term function
        void SetSource()
        {
            volScalarField yPos = T.mesh().C().component(vector::Y).ref();
            volScalarField xPos = T.mesh().C().component(vector::X).ref();
            volScalarField zPos = T.mesh().C().component(vector::Z).ref();

            // double x=32;
            // double y=7;
            // double z=42; 
            double x=30, y=-7, z=37;

            forAll(S, counter)
            {
                double distSquared = std::pow(xPos[counter]-x, 2)+std::pow(yPos[counter]-y, 2)+std::pow(zPos[counter]-z, 2);
                // S[counter] = 1e2*Foam::exp(-1/(400-distSquared));

                if (distSquared<400)
                {
                    S[counter] = 1e2*Foam::exp(-1/(400-distSquared));
                }
            }

            ITHACAstream::exportSolution(S, "1", "source", "S");
        }

};

void offline_stage(bunny& example, bunny& FOM_test);
void online_stage(bunny& example, bunny& FOM_test);

int main(int argc, char* argv[])
{
    // load stage to perform
    argList::addOption("stage", "offline", "Perform offline stage");
    argList::addOption("stage", "online", "Perform online stage");
    // add options for parallel run
    HashTable<string> validParOptions;
    validParOptions.set
    (
        "stage",
        "offline"
    );
    validParOptions.set
    (
        "stage",
        "online"
    );
    Pstream::addValidParOptions(validParOptions);
    // Construct the tutorial object
    bunny example(argc, argv);
    /// Create a new instance of the FOM problem for testing purposes
    bunny FOM_test(argc, argv);

    if (example._args().get("stage").match("offline"))
    {
        // perform the offline stage, extracting the modes from the snapshots'
        // dataset corresponding to parOffline
        offline_stage(example, FOM_test);
    }
    else if (example._args().get("stage").match("online"))
    {
        // load precomputed modes and reduced matrices
        offline_stage(example, FOM_test);
        // perform online solve with respect to the parameters in parOnline
        online_stage(example, FOM_test);
    }
    else
    {
        Info << "Pass '-stage offline', '-stage online'" << endl;
    }
    exit(0);
}

void offline_stage(bunny& example, bunny& FOM_test)
{
    // Read some parameters from file
    ITHACAparameters* para = ITHACAparameters::getInstance(example._mesh(),
                             example._runTime());
    int NmodesTout = para->ITHACAdict->lookupOrDefault<int>("NmodesTout", 15);
    int NmodesTproj = para->ITHACAdict->lookupOrDefault<int>("NmodesTproj", 10);
    // Set the number of parameters
    example.Pnumber = 1;
    example.Tnumber = 1;
    // Set the parameters
    example.setParameters();
    // Set the parameter ranges, in all the subdomains the diffusivity varies between
    // 0.001 and 0.1
    example.mu_range(0, 0) = 1;
    example.mu_range(0, 1) = 1;
    // Generate the Parameters
    example.genRandPar(1);
    // Set the size of the list of values that are multiplying the affine forms
    example.theta.resize(1);
    // Set the source term
    example.SetSource();
    // Compute the diffusivity field for each subdomain
    // example.compute_nu();
    // // Assemble all the operators of the affine decomposition
    // example.assemble_operator();
    // Perform an Offline Solve
    example.offlineSolve();
    // Perform a POD decomposition and get the modes
    // ITHACAPOD::getModes(example.Tfield, example.Tmodes, example._T().name(),
    //                     example.podex, 0, 0,
    //                     NmodesTout);
    // Perform the Galerkin projection onto the space spanned by the POD modes
    // example.project(NmodesTproj);
    // FOM_test.offline = false;
    // FOM_test.Pnumber = 9;
    // FOM_test.Tnumber = 50;
    // // Set the parameters
    // FOM_test.setParameters();
    // // Set the parameter ranges, in all the subdomains the diffusivity varies between
    // // 0.001 and 0.1
    // FOM_test.mu(0, 0) = Eigen::MatrixXd::Ones(9, 1) * 0.001;
    // FOM_test.mu_range.col(1) = Eigen::MatrixXd::Ones(9, 1) * 0.1;
    // // Generate the Parameters
    // FOM_test.genRandPar(50);
    // // Set the size of the list of values that are multiplying the affine forms
    // FOM_test.theta.resize(9);
    // // Set the source term
    // FOM_test.SetSource();
    // // Compute the diffusivity field for each subdomain
    // FOM_test.compute_nu();
    // // Assemble all the operators of the affine decomposition
    // FOM_test.assemble_operator();
    // // Perform an Offline Solve
    // FOM_test.offlineSolve("./ITHACAoutput/FOMtest");
}

void online_stage(bunny& example, bunny& FOM_test)
{
    // Create a reduced object
    reducedLaplacian reduced(example);

    // Solve the online reduced problem some new values of the parameters
    for (int i = 0; i < FOM_test.mu.rows(); i++)
    {
        reduced.solveOnline(FOM_test.mu.row(i));
    }

    // Reconstruct the solution and store it into Reconstruction folder
    reduced.reconstruct("./ITHACAoutput/Reconstruction");
    // Compute the error on the testing set
    Eigen::MatrixXd error = ITHACAutilities::errorL2Rel(FOM_test.Tfield,
                            reduced.Trec);
}
//--------
/// \dir 02thermalBlock Folder of the turorial 2
/// \file
/// \brief Implementation of a tutorial of a steady heat transfer problem

/// \example 02thermalBlock.C
/// \section intro_thermal Introduction to tutorial 2
/// The problems consists of a thermal block with a source term.
/// The problem equations are:
/// \f[
/// \nabla \cdot (k \nabla T) = S
/// \f]
/// where \f$k\f$ is the diffusivity, \f$T\f$ is the temperature and \f$S\f$ is the source term.
/// The problem discretised and formalized in matrix equation reads:
/// \f[
/// AT = S
/// \f]
/// where \f$A\f$ is the matrix of interpolation coefficients, \f$T\f$ is the vector of unknowns
/// and \f$S\f$ is the vector representing the source term.
/// The domain is subdivided in 9 different parts and each part has parametrized diffusivity. See the image below for a clarification.
/// \image html drawing.png
/// Both the full order and the reduced order
/// problem are solved exploiting the parametric affine decomposition of the differential operators:
/// \f[
/// A = \sum_{i=1}^N \theta_i(\mu) A_i
///  \f]
/// For the operations performed by each command check the comments in the source 02thermalBlock.C file.
///
/// \section code A look under the code
///
/// In this section are explained the main steps necessary to construct the tutorial N°2
///
/// \subsection header The necessary header files
///
/// First of all let's have a look to the header files that needs to be included and what they are responsible for:
///
/// The standard C++ header for input/output stream objects:
///
/// \dontinclude 02thermalBlock.C
///
/// \skip iostream
/// \until >
///
/// The OpenFOAM header files:
///
/// \skipline fvCFD
/// \until Time.H
///
/// The header file of ITHACA-FV necessary for this tutorial
///
/// \skipline ITHACAPOD
/// \until ITHACAuti
///
/// The Eigen library for matrix manipulation and linear and non-linear algebra operations:
///
/// \line Eigen
///
/// And we define some mathematical constants and include the standard header for common math operations:
///
/// \skipline MATH_DEF
/// \until cmath
///
/// \subsection classtuto02 Implementation of the bunny class
///
/// Then we can define the bunny class as a child of the laplacianProblem class
///
/// \skipline bunny
/// \until {}
///
/// The members of the class are the fields that needs to be manipulated during the
/// resolution of the problem
///
/// Inside the class it is defined the offline solve method according to the
/// specific parametrized problem that needs to be solved.
///
/// \skipline void
/// \until {
///
/// If the offline solve has already been performed than read the existing snapshots
///
/// \skipline if
/// \until
/// }
///
/// else perform the offline solve where a loop over all the parameters is performed:
/// \skipline for
/// \until }
///
/// a 0 internal constant value is assigned before each solve command with the lines
///
/// \skipline assignIF
///
/// and the solve operation is performed, see also the laplacianProblem class for the definition of the methods
///
/// \skipline truthSolve
///
/// The we need also to implement a method to set/define the source term that may be problem dependent.
/// In this case the source term is defined with a hat function:
///
///
/// \skipline SetSource
/// \until }
/// \skipline }
///
/// Define by:
///
/// \f[ S = \sin(\frac{\pi}{L}\cdot x) + \sin(\frac{\pi}{L}\cdot y) \f]
///
/// where \f$L\f$ is the dimension of the thermal block which is equal to 0.9.
///
/// \image html hat.jpg
///
/// With the following is defined a method to set compute the parameter of the affine expansion:
///
/// \skipline compute_nu
/// \until {
///
/// The list of parameters is resized according to number of parametrized regions
/// \skipline nu_list
///
/// The nine different volScalarFields to identify the viscosity in each domain are initialized:
///
/// \skipline volScalarField
/// \until nu9
///
/// and the 9 different boxes are defined:
///
/// \skipline Box1
/// \until Box9
///
/// and for each of the defined boxes the relative diffusivity field is set to 1 inside the box and remain 0 elsewhere:
///
/// \skipline ITHACA
/// \until Box9
///
///  See also the ITHACAutilities::setBoxToValue for more details.
///
/// The list of diffusivity fields is set with:
///
/// \skipline nu_list
/// \until }
///
/// \subsection main Definition of the main function
///
/// Once the bunny class is defined the main function is defined,
/// an example of type bunny is constructed:
///
/// \skipline argv)
///
/// the number of parameter is set:
///
/// \skipline Pnumber
/// \skipline setParameters
///
/// the range of the parameters is defined:
///
/// \skipline mu_range
/// \skipline mu_range
///
/// and 500 random combinations of the parameters are generated:
///
/// \skipline genRandPar
///
/// the size of the list of values that are multiplying the affine forms is set:
///
/// \skipline theta.resize
///
/// the source term is defined, the compute_nu and assemble_operator functions are called
///
/// \skipline .SetSource
/// \skipline .compute_nu
/// \skipline .assemble_operator
///
/// then the Offline full order Solve is performed:
///
/// \skipline offlineSolve
///
/// Once the Offline solve is performed the modes ar obtained using the ITHACAPOD::getModes function:
///
/// \skipline ITHACAPOD
///
/// and the projection is performed onto the POD modes using 10 modes
///
/// \skipline .project(NmodesTproj)
///
/// Once the projection is performed we can construct a reduced object:
///
/// \skipline reducedLaplacian
///
/// and solve the reduced problem for some values of the parameters:
///
/// \skipline for
/// \until }
///
/// Finally, once the online solve has been performed we can reconstruct the solution:
///
/// \skipline reconstruct
///
/// \section plaincode The plain program
/// Here there's the plain code
///
