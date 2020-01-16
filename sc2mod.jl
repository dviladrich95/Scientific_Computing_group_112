module ViladrichCollinHW02()

using Pandas
using Triangulate
using PyPlot
using ExtendableSparse
using SparseArrays
using Printf

	function plot(Plotter::Module,u,pointlist, trianglelist)
	    cmap="coolwarm"
	    levels=10
	    t=transpose(trianglelist.-1)
	    x=view(pointlist,1,:)
	    y=view(pointlist,2,:)
	    ax=Plotter.matplotlib.pyplot.gca()
	    ax.set_aspect(1)
	    #  tricontour/tricontourf takes triangulation as argument
	    Plotter.tricontourf(x,y,t,u,levels=levels,cmap=cmap)
	    PyPlot.colorbar(shrink=0.5)
	    Plotter.tricontour(x,y,t,u,levels=levels,colors="k")
	end

	function plotpair1(Plotter::Module, triin, triout;voronoi=nothing)
	    if Triangulate.ispyplot(Plotter)
	        PyPlot=Plotter
	        PyPlot.clf()
	        PyPlot.subplot(121)
	        PyPlot.title("In")
	        Triangulate.plot(PyPlot,triin)
	        PyPlot.subplot(122)
	        PyPlot.title("Out")
	        Triangulate.plot(PyPlot,triout,voronoi=voronoi)

	    end
	end


	function plotpair(Plotter::Module,u,triout)
	    if ispyplot(Plotter)
	        PyPlot=Plotter
	        PyPlot.clf()
	        PyPlot.subplot(121)
	        PyPlot.title("Grid")
	        Triangulate.plot(PyPlot,triout)
	        PyPlot.subplot(122)
	        PyPlot.title("Solution")
	        plot(PyPlot,u,triout.pointlist, triout.trianglelist)

	    end
	end

	function hmin(pointlist,trianglelist)
	    num_edges_per_cell=3
	    local_edgenodes=zeros(Int32,2,3)
	    local_edgenodes[1,1]=2
	    local_edgenodes[2,1]=3

	    local_edgenodes[1,2]=3
	    local_edgenodes[2,2]=1

	    local_edgenodes[1,3]=1
	    local_edgenodes[2,3]=2

	    h=10000.0
	    ntri=size(trianglelist,2)
	    for itri=1:ntri
	        for iedge=1:num_edges_per_cell
	            k=trianglelist[local_edgenodes[1,iedge],itri]
	            l=trianglelist[local_edgenodes[2,iedge],itri]
	            dx=pointlist[1,k]-pointlist[1,l]
	            dy=pointlist[2,k]-pointlist[2,l]
	            h=min(h,dx^2+dy^2)
	        end
	    end
	    return sqrt(h)
	end



	function compute_edge_matrix(itri, pointlist, trianglelist)
	    i1=trianglelist[1,itri];
	    i2=trianglelist[2,itri];
	    i3=trianglelist[3,itri];

	    V11= pointlist[1,i2]- pointlist[1,i1];
	    V12= pointlist[1,i3]- pointlist[1,i1];

	    V21= pointlist[2,i2]- pointlist[2,i1];
	    V22= pointlist[2,i3]- pointlist[2,i1];

	    det=V11*V22 - V12*V21;
	    vol=0.5*det
	    return (V11,V12,V21,V22,vol)
	end


	function  compute_local_stiffness_matrix!(local_matrix,itri, pointlist,trianglelist)

	    (V11,V12,V21,V22,vol)=compute_edge_matrix(itri, pointlist, trianglelist)

	    fac=0.25/vol

	    local_matrix[1,1] = fac * (  ( V21-V22 )*( V21-V22 )+( V12-V11 )*( V12-V11 ) );
	    local_matrix[2,1] = fac * (  ( V21-V22 )* V22          - ( V12-V11 )*V12 );
	    local_matrix[3,1] = fac * ( -( V21-V22 )* V21          + ( V12-V11 )*V11 );

	    local_matrix[2,2] =  fac * (  V22*V22 + V12*V12 );
	    local_matrix[3,2] =  fac * ( -V22*V21 - V12*V11 );

	    local_matrix[3,3] =  fac * ( V21*V21+ V11*V11 );

	    local_matrix[1,2] = local_matrix[2,1];
	    local_matrix[1,3] = local_matrix[3,1];
	    local_matrix[2,3] = local_matrix[3,2];
	    return vol
	end

	function  assemble!(matrix, # Global stiffness matrix
	                    rhs,    # Right hand side vector
	                    frhs::Function, # Source/sink function
	                    gbc::Function,  # Boundary condition function
	                    pointlist,
	                    trianglelist,
	                    segmentlist)
	    num_nodes_per_cell=3;
	    ntri=size(trianglelist,2)
	    vol=0.0
	    local_stiffness_matrix= [ 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0  0.0  0.0 ]
	    local_massmatrix= [ 2.0 1.0 1.0; 1.0 2.0 1.0; 1.0  1.0  2.0 ]
	    local_massmatrix./=12.0
	    rhs.=0.0

	    # Main part
	    for itri in 1:ntri
	        vol=compute_local_stiffness_matrix!(local_stiffness_matrix,itri, pointlist,trianglelist);
	        for i  in 1:num_nodes_per_cell
	            for j in 1:num_nodes_per_cell
	                k=trianglelist[j,itri]
	                x=pointlist[1,k]
	                y=pointlist[2,k]
	                rhs[trianglelist[i,itri]]+=vol*local_massmatrix[i,j]*frhs(x,y)
	                matrix[trianglelist[i,itri],trianglelist[j,itri]]+=local_stiffness_matrix[i,j]
	            end
	        end
	    end
	    # Assemble penalty terms for Dirichlet boundary conditions
	    penalty=1.0e30
	    nbface=size(segmentlist,2)
	    for ibface=1:nbface
	        for i=1:2
	            k=segmentlist[i,ibface]
	            matrix[k,k]+=penalty
	            x=pointlist[1,k]
	            y=pointlist[2,k]
	            rhs[k]+=penalty*gbc(x,y)
	        end
	    end
	end


	function discretize(n,refinement)
		r = 1
		x = zeros(n,2) #list of discretization points
		x[1,:] = [0 r] #starting point x_1

		alpha = 2 * pi / n #rotation angle for every discretization point

		for i = 2:n
			x[i,1] = cos(alpha) * x[i-1,1] - sin(alpha) * x[i-1,2]
			x[i,2] = sin(alpha) * x[i-1,1] + cos(alpha) * x[i-1,2]
		end

		input= "Dcva"*refinement

		triin=Triangulate.TriangulateIO()
		triin.pointlist=x'
		(triout, vorout)=triangulate(input, triin)
		#plotpair(PyPlot,triin,triout)

		return triout

	end


	function norms(u,pointlist,trianglelist)
	    l2norm=0.0
	    h1norm=0.0
	    num_nodes_per_cell=3
	    ntri=size(trianglelist,2)
	    local_stiffness_matrix= [ 0.0 0.0 0.0; 0.0 0.0 0.0; 0.0  0.0  0.0 ]
	    local_mass_matrix= [ 2.0 1.0 1.0; 1.0 2.0 1.0; 1.0  1.0  2.0 ]
	    local_mass_matrix./=12.0
	    for itri=1:ntri
	        vol=compute_local_stiffness_matrix!(local_stiffness_matrix,itri, pointlist,trianglelist);
	        for i  in 1:num_nodes_per_cell
	            for j in 1:num_nodes_per_cell
	                uij=u[trianglelist[j,itri]]*u[trianglelist[i,itri]]
	                l2norm+=uij*vol*local_mass_matrix[j,i]
	                h1norm+=uij*local_stiffness_matrix[j,i]
	            end
	        end
	    end
	    return (sqrt(l2norm),sqrt(h1norm));
	end





	function funtask1(n,r) #n=number of discretization points, r=radius of circle
		x = zeros(n,2) #list of discretization points
		x[1,:] = [0 r] #starting point x_1

		alpha = 2 * pi / n #rotation angle for every discretization point

		for i = 2:n
			x[i,1] = cos(alpha) * x[i-1,1] - sin(alpha) * x[i-1,2]
			x[i,2] = sin(alpha) * x[i-1,1] + cos(alpha) * x[i-1,2]
		end

		precision_array = [	"0.025"
							"0.00625"
							"0.0015625"
							"0.000390625"
							"0.00009765625"
							"0.0000244140625"
							"0.000006103515625" ]
		refinement_level = zeros(7)
		smallest_edge_length = zeros(7)
		number_of_triangles = zeros(7)
		number_of_vertices = zeros(7)



		for i=1:7

			input= "Dcva"*precision_array[i]

			triin=Triangulate.TriangulateIO()
			triin.pointlist=x'
			(triout, vorout)=triangulate(input, triin)
			#plotpair1(PyPlot,triin,triout)
			refinement_level[i]=i
			smallest_edge_length[i]=hmin(triout.pointlist,triout.trianglelist)
			number_of_triangles[i]=length(triout.trianglelist[1,:])
			number_of_vertices[i]=length(triout.pointlist[1,:])

		end

		area_polygon = (x[1,1]*x[2,2] - x[1,2]*x[2,1]) * n / 2
		approximation_error = pi*r^2 - area_polygon

		result = string("Approximation by a polygon with ",n," approximation points:\n")
		for i=1:7
			result=result*string(	"Refinement level: ",refinement_level[i],
					" | Smallest edge length: ",smallest_edge_length[i],
					" | Number of triangles: ",number_of_triangles[i],
					" | Number of vertices: ",number_of_vertices[i],"\n")
		end

		result=result*string("Approximation error of polygon: ",approximation_error,"\n")

		return result



	end

	function task1()
		result="\n"
		for i=1:10
			result=result*string(funtask1(i*10,1),"\n")
		end
		print(result)
	end

#=


Discussion Task 2.
To compare the exact solution with the approximation by the mesh, we first note
that the exact solution is a paraboliod, and thus has radial symmetry and
decreases steadily from 1/4 for r=0 to 0 for r=1. For coarse outlines of the
polygonal border the FEM approximation cannot reproduce the radial symmetry of
the exact solution even for high refinement levels. On the other hand, coarse
mesh generation leads to irregular level curves throughout the interior of the
mesh and become more pronounced in the center region where the FEM approximation
is farther away from the boundary conditions, where the values of FEM and
exact solution coincide by construction.


=#



	function task2()
		error_table = zeros(10,10)
		for polyline_index=1:10
			n_polyline=polyline_index+2#2^polyline_index+1
			for refinement=1:10
				mesh_size=0.1*2.0^(-refinement)
				print(mesh_size)
				triout = discretize(n_polyline,string(mesh_size))

				n=size(triout.pointlist,2)
				frhs(x,y)=1
			    gbc(x,y)=0
			    matrix=spzeros(n,n)
			    rhs=zeros(n)
			    assemble!(matrix,rhs,frhs,gbc,triout.pointlist,triout.trianglelist, triout.segmentlist)
			    sol=matrix\rhs

			    u(x,y)=-1/4*(x^2+y^2)+(1/4)

			    u_vector=zeros(n)
			    error=0
			    for i=1:n
			    	error += abs(u(triout.pointlist[1,i],triout.pointlist[2,i])-sol[i])
			    	u_vector[i]=u(triout.pointlist[1,i],triout.pointlist[2,i])
			    end
			    norm_vector=u_vector-sol
			    @show norms(norm_vector,triout.pointlist,triout.trianglelist)
				@show avg_error=error/n
				@show avg_log_error=(avg_error)

				error_table[polyline_index,refinement]=avg_log_error
				PyPlot.clf()

			    plotpair(PyPlot,sol,triout)
				#savefig("plotpair$n_polyline$refinement.png")
				PyPlot.clf()

			end
		end
		pcolormesh(error_table)
		colorbar()
		#savefig("log_error_heatmap.png")
		PyPlot.clf()
		for i in 1:10
			scatter(collect(1:10),error_table[i,:])
		end
		savefig("error_scatter.png")
	end




end
