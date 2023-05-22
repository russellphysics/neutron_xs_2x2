import matplotlib
import matplotlib.pyplot as plt
import h5py
import argparse
import numpy as np
import glob
import twoBytwo_defs
import auxiliary

file_dir='/global/cfs/cdirs/dune/www/data/2x2/simulation/productions/MiniRun3_1E19_RHC/MiniRun3_1E19_RHC.convert2h5.withMinerva/EDEPSIM_H5/'

#### signal = elastic scatter fiducialized proton via neutron and fiducialized nu vertex


def fiducial_nu(spill_id, ghdr, seg, vert, traj, output):
    vert_mask = vert['spillID']==spill_id
    nu_vert = vert[vert_mask]
    for v in nu_vert:
        vert_loc = [nu_vert['x_vert'],nu_vert['y_vert'],nu_vert['z_vert']]
        vert_loc = auxiliary.np_array_of_array_to_flat_list(vert_loc)    
        tpc_fv = twoBytwo_defs.tpc_vertex(vert_loc)
        in_tpc=False
        for key in tpc_fv.keys():
            if tpc_fv[key]==True: in_tpc=True
        if in_tpc==False: continue

        traj_mask=traj['vertexID']==v['vertexID']
        final_states=traj[traj_mask]

        track_ids_seen=set()
        for fs in final_states:
            if fs['parentID']!=-1: continue

            track_id = fs['trackID']
            if track_id in track_ids_seen: continue

            pdg=fs['pdgId']
            track_id_set = auxiliary.same_pdg_connected_trajectories(pdg, track_id, \
                                                                     final_states, \
                                                                     traj, ghdr)
            track_ids_seen.update(track_id_set)

            total_edep, contained_edep, total_length, contained_length=[0. for i in range(4)]
            for tis in track_id_set:
                total_edep+=auxiliary.total_energy(pdg, tis, traj, seg)
                contained_edep+=auxiliary.fv_contained_energy(pdg, tis, traj, seg)
                total_length+=auxiliary.total_length(pdg, tis, traj, seg)
                contained_length += auxiliary.fv_contained_length(pdg, tis, traj, seg)
            output[(spill_id, int(v['vertexID']), track_id)]=dict(
                pdg=int(pdg),
                total_edep=float(total_edep),
                contained_edep=float(contained_edep),
                total_length=float(total_length),
                contained_length=float(contained_length) )
                
                                                                     

def construct_dict(spill_id, ghdr, seg, vert, traj, output):
    seg_proton_mask = seg['pdgId']==2212
    proton_seg = seg[seg_proton_mask]
    traj_proton_mask = traj['pdgId']==2212
    proton_traj=traj[traj_proton_mask]
    for pt in proton_traj:

        # proton start/end points in fiducial volume
        proton_start=pt['xyz_start']
        if twoBytwo_defs.fiducialized_vertex(proton_start)==False: continue
        if twoBytwo_defs.fiducialized_vertex(pt['xyz_end'])==False: continue

        proton_energy=auxiliary.total_energy(pt['pdgId'], pt['trackID'], traj, seg)
        if proton_energy==0.: continue
        proton_length=auxiliary.total_length(pt['pdgId'], pt['trackID'], traj, seg)
        if proton_length==0.: continue

        # find all primaries
        traj_mask=traj['vertexID']==pt['vertexID']
        final_states=traj[traj_mask]
        primaries=[]
        for fs in final_states:
            if fs['parentID']==-1: primaries.append(int(fs['pdgId']))
        
        # neutrino vertex position corresponding to fiducialized proton
        vert_mask = vert['vertexID']==pt['vertexID']
        nu_vert = vert[vert_mask]
        vert_loc = [nu_vert['x_vert'],nu_vert['y_vert'],nu_vert['z_vert']]
        vert_loc = auxiliary.np_array_of_array_to_flat_list(vert_loc)

        # find nu interaction location
        placement='o'
        tpc_fv = twoBytwo_defs.tpc_vertex(vert_loc)
        in_tpc=False
        for key in tpc_fv.keys():
            if tpc_fv[key]==True:
                placement=str(key)
        if placement=='o':
            if twoBytwo_defs.minerva_vertex(vert_loc)[0]:
                if twoBytwo_defs.minerva_vertex(vert_loc)[1]:
                    placement='u'
                else:
                    placement='d'

        nu_proton_dt = nu_vert['t_vert'][0] - pt['t_start']
                    
        nu_proton_distance = np.sqrt( (proton_start[0]-vert_loc[0])**2+
                                      (proton_start[1]-vert_loc[1])**2+
                                      (proton_start[2]-vert_loc[2])**2)

        parent_mask = traj['trackID']==pt['parentID']
        parent_pdg = traj[parent_mask]['pdgId']
        if pt['parentID']==-1:
            ghdr_mask=ghdr['vertexID']==pt['vertexID']
            parent_pdg=ghdr[ghdr_mask]['nu_pdg']

        output[(spill_id, pt['vertexID'], pt['trackID'])]=dict(
            vertex=placement,
            nu_p_dt=float(nu_proton_dt),
            nu_p_d=float(nu_proton_distance),
            p_vis_e=float(proton_energy),
            p_len=float(proton_length),
            parent_pdg=int(parent_pdg[0]),
            primaries=primaries
        )
    return



def main(sim_file, save):
    nu=dict()
    d=dict()
    file_ctr=0
    for sim_file in glob.glob(file_dir+'*.EDEPSIM.h5'):
        file_ctr+=1
        if file_ctr>100: break
        print('FILE # ',file_ctr)
        sim_h5 = h5py.File(sim_file,'r')    
        unique_spill = np.unique(sim_h5['trajectories']['spillID'])
        for spill_id in unique_spill:
            ghdr, gstack, traj, vert, seg = auxiliary.get_spill_data(sim_h5, spill_id)    
            construct_dict(spill_id, ghdr, seg, vert, traj, d)
            fiducial_nu(spill_id, ghdr, seg, vert, traj, nu)
    auxiliary.save_dict_to_json(d, 'neutron_tof',True)
    auxiliary.save_dict_to_json(nu, 'fv_nu', True)
    print('JSON made')

        

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--sim_file',default=None, required=False, type=str,\
                        help='''string corresponding to the path of the edepsim file ''')
    parser.add_argument('-s','--save',default=True, type=bool, \
                        help='''save plot(s) to PNG; otherwise, show in screen ''')
    args = parser.parse_args()
    main(**vars(args))
