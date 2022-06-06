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
#include <exploragram/optimal_transport/optimal_transport_2d.h>
#include <exploragram/optimal_transport/sampling.h>

namespace {
    using namespace GEO;
}


int main(int argc, char** argv) {
    using namespace GEO;

    GEO::initialize();
	CmdLine::import_arg_group("standard");
	CmdLine::import_arg_group("pre");
      	CmdLine::import_arg_group("remesh");
      	CmdLine::import_arg_group("post");
        CmdLine::import_arg_group("algo");
        CmdLine::import_arg_group("opt");
	CmdLine::import_arg_group("co3ne");
	CmdLine::import_arg_group("stat");
        std::string mesh1_filename = "rectangle.stl";
        std::string mesh2_filename = "square.stl";

        Mesh M1;
        Mesh M2;
        Mesh M2_samples;
        Mesh M3;
        MeshIOFlags flags;
        flags.set_element(MESH_CELLS);
        flags.set_attribute(MESH_CELL_REGION);
        mesh_load(mesh1_filename, M1,flags);
        mesh_load(mesh2_filename, M2,flags);
       	//recenter_mesh(M1,M2);
 	//rescale_mesh(M1,M2);

	CentroidalVoronoiTesselation CVT(&M2, 0, "NN");
	int npoints=1000;
	CVT.compute_initial_sampling(npoints); // Warning: Did put all the points in the same triangle 
	CVT.Lloyd_iterations(100);
	CVT.Newton_iterations(100); 
	M2_samples.vertices.assign_points(
            CVT.embedding(0), CVT.dimension(), CVT.nb_points()
        );
        std::cout<<"number of samples"<<std::endl;
	std::cout<<M2_samples.vertices.nb()<<std::endl;

        Logger::div("Optimal transport");
        //M1.vertices.set_dimension(4);
        OptimalTransportMap2d OTM(&M1);
        OTM.set_points(
            M2_samples.vertices.nb(), M2_samples.vertices.point_ptr(0)
        );
        OTM.set_epsilon(0.0001);
        OTM.optimize(1000);//Warning: All the points are colinear and segfault
        double *centroids=new double[3*npoints];        
	//OTM.compute_Laguerre_centroids(centroids); crashes 
	//OTM.get_RVD(M3);  

	delete [] centroids;
        

  
	

    return 0;
}

