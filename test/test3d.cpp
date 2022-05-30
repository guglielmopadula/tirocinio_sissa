/*
 *  Copyright (c) 2012-2014, Bruno Levy
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *  this list of conditions and the following disclaimer in the documentation
 *  and/or other materials provided with the distribution.
 *  * Neither the name of the ALICE Project-Team nor the names of its
 *  contributors may be used to endorse or promote products derived from this
 *  software without specific prior written permission.
 * 
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  If you modify this software, you should include a notice giving the
 *  name of the person performing the modification, the date of modification,
 *  and the reason for such modification.
 *
 *  Contact: Bruno Levy
 *
 *     Bruno.Levy@inria.fr
 *     http://www.loria.fr/~levy
 *
 *     ALICE Project
 *     LORIA, INRIA Lorraine, 
 *     Campus Scientifique, BP 239
 *     54506 VANDOEUVRE LES NANCY CEDEX 
 *     FRANCE
 *
 */

#include <geogram/basic/common.h>
#include <geogram/basic/logger.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/basic/stopwatch.h>
#include <geogram/basic/file_system.h>
#include <geogram/basic/process.h>
#include <geogram/basic/progress.h>
#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_io.h>
#include <geogram/mesh/mesh_tetrahedralize.h>
#include <geogram/voronoi/CVT.h>

#include <exploragram/optimal_transport/optimal_transport_3d.h>
#include <exploragram/optimal_transport/sampling.h>

namespace {
    using namespace GEO;


    /**
     * \brief Loads a volumetric mesh.
     * \details If the specified file contains a surface, try to
     *  tesselate it. If the surface has self-intersections, try to
     *  remove them.
     * \param[in] filename the name of the file
     * \param[out] M the mesh
     * \retval true if the file was successfully loaded
     * \retval false otherwise
     */
    bool load_volume_mesh(const std::string& filename, Mesh& M) {
        MeshIOFlags flags;
        flags.set_element(MESH_CELLS);
        flags.set_attribute(MESH_CELL_REGION);

        if(!mesh_load(filename, M, flags)) {
            return 1;
        }
        if(!M.cells.are_simplices()) {
            Logger::err("I/O") << "File "
                               << filename
                               << " should only have tetrahedra" << std::endl;
            return false;
        }
        if(M.cells.nb() == 0) {
            Logger::out("I/O") << "File "
                               << filename
                               << " does not contain a volume" << std::endl;
            Logger::out("I/O") << "Trying to tetrahedralize..." << std::endl;
            if(!mesh_tetrahedralize(M,true,false)) {
                return false;
            }
        }
        return true;
    }
}

int main(int argc, char** argv) {
    using namespace GEO;

    GEO::initialize();

    try {
        
        std::vector<std::string> filenames;

        CmdLine::import_arg_group("standard");
        CmdLine::import_arg_group("algo");
        CmdLine::import_arg_group("opt");
       /*
        CmdLine::declare_arg("nb_pts", 10000, "number of points");
        CmdLine::declare_arg("nb_iter", 10000, "number of iterations for OTM");
        CmdLine::declare_arg("RDT", false, "save regular triangulation");
        CmdLine::declare_arg_group(
            "RVD", "RVD output options", CmdLine::ARG_ADVANCED
        );
        CmdLine::declare_arg("RVD", false, "save restricted Voronoi diagram");
        CmdLine::declare_arg(
            "RVD_iter", false, "save restricted Voronoi diagram at each iteration"
        );
        CmdLine::declare_arg(
            "RVD:borders_only", false, "save only border of RVD"
        );        
        CmdLine::declare_arg(
            "RVD:integration_simplices", true, "export RVD as integration simplices"
        );        
        
        CmdLine::declare_arg("multilevel", true, "use multilevel algorithm");
        CmdLine::declare_arg("BRIO", true, 
                             "use BRIO reordering to compute the levels"
        );
        CmdLine::declare_arg("ratio", 0.125, "ratio between levels");
        CmdLine::declare_arg(
            "epsilon", 0.001, "relative measure error in a cell"
        );
        CmdLine::declare_arg(
            "lock", true, "Lock lower levels when sampling shape"
        );
        CmdLine::declare_arg(
            "fitting_degree", 4, "degree for interpolating weights"
        );
        CmdLine::declare_arg(
            "project", true, "project sampling on border"
        );
        CmdLine::declare_arg(
            "feature_sensitive", true, "attempt to recover hard edges"
        );
        CmdLine::declare_arg(
            "singular", false, "compute and save singular surface"
        );
        CmdLine::set_arg("algo:delaunay", "BPOW");
        CmdLine::declare_arg(
            "recenter", true, "recenter target onto source mesh"
        );
        CmdLine::declare_arg(
            "rescale", true, "rescale target to match source volume"
        );
        CmdLine::declare_arg(
            "density_min", 1.0, "min density in first mesh"
        );
        CmdLine::declare_arg(
            "density_max", 1.0, "max density in first mesh"
        );
        CmdLine::declare_arg(
            "density_function", "x", "used function for density"
        );
        CmdLine::declare_arg(
            "density_distance_reference", "",
            "filename of the reference surface"
        );
        CmdLine::declare_arg(
            "out", "morph.tet6", "output filename"
        );
        */
        std::string mesh1_filename = "sphere.stl";
        std::string mesh2_filename = "sphere.stl";
        Mesh M1;
        Mesh M2;
        Mesh M2_samples;
        /**
        if(!load_volume_mesh(mesh1_filename, M1)) {
            return 1;
        }
        
        if(!load_volume_mesh(mesh2_filename, M2)) {
            return 1;
        }
	**/
	MeshIOFlags flags;
        flags.set_element(MESH_CELLS);
        flags.set_attribute(MESH_CELL_REGION);
        mesh_load(mesh1_filename, M1,flags);
        mesh_load(mesh2_filename, M2,flags);
        mesh_tetrahedralize(M2,true,false);
        mesh_tetrahedralize(M1,true,false);
	// TODO: distance reference...
        set_density(
            M1,
            1,
           1,
	   "x"
        );
            recenter_mesh(M1,M2);
            rescale_mesh(M1,M2);
        


        


        CentroidalVoronoiTesselation CVT(&M2, 0, "NN");

    }
    
    
    catch(const std::exception& e) {
        std::cerr << "Received an exception: " << e.what() << std::endl;
        return 1;
    }
/*
    Logger::out("") << "Everything OK, Returning status 0" << std::endl;
    */
    
    return 0;
}

