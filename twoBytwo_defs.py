import numpy as np

loc_dict={'o':'Outside Active Volume',
          'f':'LAr Fiducial Volume',
          'u':r'MINER$\nu$A Upstream',
          'd':r'MINER$\nu$A Downstrem',
          'm':r'MINER$\nu$A',
          '0':'TPC 0',
          '1':'TPC 1',
          '2':'TPC 2',
          '3':'TPC 3',
          '4':'TPC 4',
          '5':'TPC 5',
          '6':'TPC 6',
          '7':'TPC 7'}


#### geometry definitions copied straight from https://github.com/DUNE/2x2_sim/blob/main/validation/edepsim_validation.py

NDHallwidths = [1000.,550.,2000.] # cm

def tpc_bounds(i):
    """A sad little function that returns the bounds of each 2x2 tpc in one dimension.
    The dimension is chosen by i: 0, 1, 2 -> x, y, z.
    Values are taken from 2x2_sim/run-edep-sim/geometry/Merged2x2MINERvA_v2"""
            
    active_tpc_widths = [30.6, 130., 64.] # cm
            
    # The positions in cm of the center of each tpc relative to a module center.
    # There are two tpcs for each module.            
    tpcs_relative_to_module = [[-15.7,0.,0.], [15.7, 0., 0.]]

    # The positions in cm of each of the four modules, relative to the 2x2 center position.
    modules_relative_to_2x2= [[-33.5,0.,-33.5],
                              [33.5,0.,-33.5],
                              [-33.5,0.,33.5],
                              [33.5,0.,33.5]]
            
    # The position of the 2x2 center, relative to the center of the ND hall
    detector_center = [0.,52.25,0.]
        
    # Get the tpc bounds relative to the tpc center in the ith coordinates
    tpc_bounds = np.array([-active_tpc_widths[i]/2., active_tpc_widths[i]/2.])
            
    tpc_bounds_relative_to_2x2 = []
    for tpc in tpcs_relative_to_module:
        tpc_bound_relative_to_module = tpc_bounds + tpc[i]
        for module in modules_relative_to_2x2:
            bound = tpc_bound_relative_to_module + module[i]
            tpc_bounds_relative_to_2x2.append(bound)
                    
    bounds_relative_to_NDhall = np.array(tpc_bounds_relative_to_2x2) + detector_center[i]
            
    return np.unique(bounds_relative_to_NDhall, axis = 0)


def MINERvA_bounds(i):
    """A sadder littler function that returns the bounds of the MINERvA detector for a given
    dimension i: 0, 1, 2 -> x, y, z.
    For now, I take the detector to just simply be two monolithic hexagonal prisms,
    downstream and upstream of the 2x2 modules. 
    """
            
    # Taken from the gdml file.
    MINERvA_center = [0., 43., -654.865]
                        
    # From GDML file, the length of one side of the outer detector in cm
    side_length = 199.439876988864
            
    # Properties of a hexagon, given a side length
    long_diameter = 2*side_length
    short_diameter = np.sqrt(3)/2.*long_diameter 
            
    width = short_diameter 
    height = long_diameter
            
    # The detector bounds, in each dimention. The bounds of the z dimension for each prism
    # were obtained by looking at the central positions of the first and last HCal frame 
    # position for each slab. 
    detector_bounds = np.array([ 
                [[-width/2., width/2.], [-height/2., height/2.], [403.27185, 451.18895]],
                [[-width/2., width/2.], [-height/2., height/2.], [858.54105, 997.2314]]
    ])
            
    bounds_relative_to_NDhall = [] 
    for bound in detector_bounds:
        bounds_relative_to_NDhall.append(bound[i] + MINERvA_center[i])
            
    return np.unique(np.array(bounds_relative_to_NDhall), axis = 0)


def fiducialized_vertex(vert_pos):
    flag=False; x_drift_flag=False; y_vertical_flag=False; z_beam_flag=False
    for i in range(3):
        for i_bounds, bounds in enumerate(tpc_bounds(i)):
            if vert_pos[i]>bounds[0] and vert_pos[i]<bounds[1]:
                if i==0: x_drift_flag=True; break
                if i==1: y_vertical_flag=True
                if i==2: z_beam_flag=True
    if x_drift_flag==True and y_vertical_flag==True and z_beam_flag==True: flag\
=True
    return flag



def tpc_vertex(vert_pos):
    temp=[]
    for i in range(3): temp.append(tpc_bounds(i).tolist())
    tpc_fv={}
    for i in range(8): tpc_fv[i]=False
    tpc=0
    enclosed=False
    for x in range(4):
        for y in range(1):
            for z in range(2):
                if vert_pos[0]>temp[0][x][0] and vert_pos[0]<temp[0][x][1] and\
                   vert_pos[1]>temp[1][y][0] and vert_pos[1]<temp[1][y][1] and\
                   vert_pos[2]>temp[2][z][0] and vert_pos[2]<temp[2][z][1]:
                    tpc_fv[tpc]=True
                    return tpc_fv
                tpc+=1
    return tpc_fv



def particle_containment(traj, trackID):
    mask = traj['trackID']==trackID
    start=fiducialized_vertex(traj[mask]['xyz_start'][0].tolist())
    end=fiducialized_vertex(traj[mask]['xyz_end'][0].tolist())
    if start==True and end==True: return 'fc' # fully contained
    elif (start==True and end==False) or (start==False and end==True): return 'pc' # partially contained
    else: return 'tg' # through going
    


def minerva_vertex(vert_pos):
    upstream=False
    flag=False; x_flag=False; y_flag=False; z_upstream_flag=False; z_downstream_flag=False
    for i in range(3):
        ctr=0
        for i_bounds, bounds in enumerate(MINERvA_bounds(i)):
            if i==0 and vert_pos[i]>bounds[0] and vert_pos[i]<bounds[1]:
                x_flag=True
            if i==1 and vert_pos[i]>bounds[0] and vert_pos[i]<bounds[1]:
                y_flag=True
            if i==2 and ctr==0 and \
               vert_pos[i]>bounds[0] and vert_pos[i]<bounds[1]:
                z_upstream_flag=True
            if i==2 and ctr==1 and \
               vert_pos[i]>bounds[0] and vert_pos[i]<bounds[1]:
                z_downstream_flag=True
            ctr+=1
    if x_flag==True and y_flag==True and z_upstream_flag==True:
        flag=True; upstream=True
    elif x_flag==True and y_flag==True and z_downstream_flag==True:
        flag=True; upstream=False
    return (flag, upstream)
                                          


def fiducialized_particle_origin(traj, vert_id):
    traj_vert_mask = traj['vertexID']==vert_id
    final_states = traj[traj_vert_mask]
    for fs in final_states:
        if fiducialized_vertex(fs['xyz_start'])==True:
            return True
    return False
