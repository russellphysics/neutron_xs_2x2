import matplotlib
import matplotlib.pyplot as plt
import h5py
import argparse
import numpy as np
import twoBytwo_defs
import auxiliary
import glob

file_dir='/global/cfs/cdirs/dune/www/data/2x2/simulation/productions/MiniRun3_1E19_RHC/MiniRun3_1E19_RHC.convert2h5.withMinerva/EDEPSIM_H5/'
#file_dir='/home/russell/DUNE/2x2/numubar_cc_0pi/'

def fiducial_volume_multiplicity(spill_id, seg, vert, traj, \
                                 fiducial_volume_dict, threshold=5.):
    # find all tracks crossing the fiducialized volume
    fiducial_track_id = set()
    for s in seg:
        if s['trackID'] in fiducial_track_id: continue
        point = [(s['x_start']+s['x_end'])/2.,
                 (s['y_start']+s['y_end'])/2.,
                 (s['z_start']+s['z_end'])/2.]
        if twoBytwo_defs.fiducialized_vertex(point):
            fiducial_track_id.add(s['trackID'])

    # remove tracks if FV contained energy below specified threshold
    final_fiducial_tracks = fiducial_track_id.copy() 
    fiducial_track_vis_e, fiducial_track_pdg, fiducial_track_status=[[] for i in range(3)]
    for t in fiducial_track_id:
        track_mask = seg['trackID']==t
        traj_mask = traj['trackID']==t
        pdg = traj[traj_mask]['pdgId'].tolist()[0]
        contained_e = auxiliary.fv_contained_energy(pdg, t, traj, seg)
        if contained_e<threshold:
            final_fiducial_tracks.remove(t)
            continue
        fiducial_track_vis_e.append(contained_e)
        fiducial_track_pdg.append(pdg)
        fiducial_track_status.append(twoBytwo_defs.particle_containment(traj,t))
        
    # find neutrino vertex IDs correspodning to identified tracks
    origin_vertex = set()
    for t in final_fiducial_tracks:
        mask = traj['trackID']==t
        origin_vertex.add(traj[mask]['vertexID'][0])

    # find nu vertex interaction location in detector hall geometry
    vertex_placement=[]
    for v in origin_vertex:
        placement='o'
        vert_mask = vert['vertexID']==v
        location = [vert[vert_mask]['x_vert'],vert[vert_mask]['y_vert'],vert[vert_mask]['z_vert']]
        location = auxiliary.np_array_of_array_to_flat_list(location)
        if twoBytwo_defs.fiducialized_vertex(location):
            placement='f' # fiducial volume
        elif twoBytwo_defs.minerva_vertex(location)[0]:
            if twoBytwo_defs.minerva_vertex(location)[1]:
                placement='u' # upstream MINERvA
            else:
                placement='d' # downstream MINERvA
        vertex_placement.append(placement)
        
    # save dict for later parsing
    fiducial_volume_dict[str(spill_id)]=dict(
        vertex=vertex_placement,
        contained_e=fiducial_track_vis_e,
        pdg=fiducial_track_pdg,
        status=fiducial_track_status
        )
    return

 

def tpc_multiplicity(spill_id, seg, vert, traj, tpc_dict, threshold=5.):

    #### if track traverses TPC, save track IDs to associated TPC index
    tpc_track_id={} 
    for i in range(8): tpc_track_id[i]=set()
    for s in seg:
        point = [(s['x_start']+s['x_end'])/2.,
                 (s['y_start']+s['y_end'])/2.,
                 (s['z_start']+s['z_end'])/2.]
        tpc_fv = twoBytwo_defs.tpc_vertex(point)
        for key in tpc_fv.keys():
            if tpc_fv[key]==True:
                tpc_track_id[key].add(s['trackID'])

    #### save visible energy deposited per TPC per track
    final_tpc_tracks = {}
    tpc_track_vis_e, tpc_track_pdg, tpc_track_status=[{} for i in range(3)]
    for key in tpc_track_id.keys(): final_tpc_tracks[key]=tpc_track_id[key].copy()
    for i in range(8): tpc_track_vis_e[i]=[]; tpc_track_pdg[i]=[]; tpc_track_status[i]=[]
    for key in tpc_track_id.keys():
        for t in tpc_track_id[key]:
            track_mask = seg['trackID']==t
            traj_mask = traj['trackID']==t
            pdg = traj[traj_mask]['pdgId'].tolist()[0]
            contained_e = auxiliary.tpc_contained_energy(pdg, t, traj, seg)
            for key_e in contained_e.keys():
                if contained_e[key_e]<threshold:
                    if t in final_tpc_tracks[key_e]: final_tpc_tracks[key_e].remove(t)
                    continue
                tpc_track_vis_e[key_e].append(contained_e[key])
                tpc_track_pdg[key_e].append(pdg)
                tpc_track_status[key_e].append(twoBytwo_defs.particle_containment(traj,t))

    # find neutrino vertex IDs correspodning to identified tracks
    origin_vertex = {}
    for i in range(8): origin_vertex[i]=set()
    for key in final_tpc_tracks.keys():
        for t in final_tpc_tracks[key]:
            mask = traj['trackID']==t
            origin_vertex[key].add(traj[mask]['vertexID'][0])
                
    #### save originating nu vertex qualitative location for tracks traversing TPCs
    vertex_placement={}
    for i in range(8):
        vertex_placement[i]=[]
        for v in origin_vertex[i]:
            placement='o'
            vert_mask = vert['vertexID']==v
            location = [vert[vert_mask]['x_vert'],
                        vert[vert_mask]['y_vert'],
                        vert[vert_mask]['z_vert']]
            location = auxiliary.np_array_of_array_to_flat_list(location)

            tpc_fv = twoBytwo_defs.tpc_vertex(location)
            in_tpc=False
            for key in tpc_fv.keys():
                if tpc_fv[key]==True:
                    placement=str(key)
                    in_tpc=True
                    vertex_placement[i].append(placement)
                    break
            if in_tpc: continue
        
            if twoBytwo_defs.minerva_vertex(location)[0]:
                if twoBytwo_defs.minerva_vertex(location)[1]:
                    placement='u' # upstream MINERvA
                else:
                    placement='d' # downstream MINERvA
            vertex_placement[i].append(placement)

    tpc_dict[str(spill_id)]=dict(
        vertex=vertex_placement,
        contained_e=tpc_track_vis_e,
        pdg=tpc_track_pdg,
        status=tpc_track_status
        )
    return



def plot_fv_mult(d, agg_vertex_placement=True, spill_vertex_placement=True, \
                 n_nu=True, contained_e=True, n_tracks=True, save=True):
    if agg_vertex_placement: plot_fv_aggregate_vertex_placement(d, save)
    if spill_vertex_placement: plot_fv_spill_vertex_placement(d, save)
    if n_nu: plot_fv_spill_nu_count(d, save)
    if contained_e: plot_fv_contained_e(d, save)
    if n_tracks: plot_fv_n_tracks(d, save)


    
def plot_tpc_mult(d, agg_vertex_placement=True, spill_vertex_placement=True, \
                  n_nu=True, contained_e=True, n_tracks=True, save=True):
    if agg_vertex_placement: plot_tpc_aggregate_vertex_placement(d, save)
    if spill_vertex_placement: plot_tpc_spill_vertex_placement(d, save)
    if n_nu: plot_tpc_spill_nu_count(d, save)
    if contained_e: plot_tpc_contained_e(d, save)
    if n_tracks: plot_tpc_n_tracks(d, save)


    
def plot_fv_aggregate_vertex_placement(d, save):
    fig, ax = plt.subplots(figsize=(6,4))
    placement = [v for key in d.keys() for v in d[key]['vertex']]
    placement_set = set(placement)
    placement_count = [(p, placement.count(p)) for p in placement_set]
    placement_fraction = [ 100*(i[1]/len(placement)) for i in placement_count]
    placement_label = [twoBytwo_defs.loc_dict[i[0]]  for i in placement_count]
    ax.pie(placement_fraction, labels=placement_label, autopct='%1.1f%%')
    ax.set_title('Aggregate Fiducial Visible Energy'+'\n'+r'$\nu$ Origin')
    if save==True: plt.savefig("fv_vertex_placement.png")
    else: plt.show()



def plot_tpc_aggregate_vertex_placement(d, save):
    fig, ax = plt.subplots(2,4,figsize=(24,8))
    a={0:(0,0),1:(0,1),2:(0,2),3:(0,3),
       4:(1,0),5:(1,1),6:(1,2),7:(1,3)}
    placement, placement_set, placement_count, placement_fraction, placement_label=[{} for i in range(5)]
    for i in range(8):
        placement[i]=[]
        placement_set[i]=set()
        for key in d.keys():
            for v in d[key]['vertex'][i]: placement[i].append(v)
        placement_set[i]=set(placement[i])
        placement_count[i]=[(p, placement[i].count(p)) for p in placement_set[i]]
        placement_fraction[i]=[100*(b[1]/len(placement[i])) for b in placement_count[i]]
        placement_label[i]=[twoBytwo_defs.loc_dict[b[0]] for b in placement_count[i]]
        ax[a[i][0]][a[i][1]].pie(placement_fraction[i], \
                                 labels=placement_label[i], autopct='%1.1f%%')
        ax[a[i][0]][a[i][1]].set_title('Aggregate TPC '+str(i)+' Visible Energy'+\
                                       '\n'+r'$\nu$ Origin')
    if save==True: plt.savefig("tpc_vertex_placement.png")
    else: plt.show()



def plot_fv_spill_vertex_placement(d, save):
    fig, ax = plt.subplots(figsize=(6,4,))
    spill_composition={}
    for key in d.keys():
        spill_composition[key]={}
        for v in d[key]['vertex']:
            flag=v
            if v=='u' or v=='d': flag='m'
            if flag not in spill_composition[key]: spill_composition[key][flag]=0            
            spill_composition[key][flag]+=1
    for a in ['f','u','m']:
        ax.hist([spill_composition[key][loc]/len(d[key]['vertex']) \
                 for key in spill_composition.keys() \
                 for loc in spill_composition[key] if loc==a], \
                histtype='step', label=twoBytwo_defs.loc_dict[a], linewidth=2)
    ax.grid(True)
    ax.legend()
    ax.set_title(r'Fiducial Visible Energy $\nu$ Origin')
    ax.set_xlabel(r'Fraction of $\nu$ Interactions Per Spill')
    ax.set_ylabel('Spill Count')
    if save==True: plt.savefig('fv_vertex_origin_composition.png')
    else: plt.show()


    
def plot_tpc_spill_vertex_placement(d, save):
    fig, ax = plt.subplots(2,4,figsize=(24,8))
    a={0:(0,0),1:(0,1),2:(0,2),3:(0,3),
       4:(1,0),5:(1,1),6:(1,2),7:(1,3)}
    for i in range(8):
        spill_composition={}
        for key in d.keys():
            spill_composition[key]={}
            for v in d[key]['vertex'][i]:
                flag=v
                if v=='u' or v=='d': flag='m'
                if v not in spill_composition[key]: spill_composition[key][flag]=0
                spill_composition[key][flag]+=1
        for b in ['f','u','m']:
            ax[a[i][0]][a[i][1]].hist([spill_composition[key][loc]/len(d[key]['vertex'][i]) \
                                       for key in spill_composition.keys() \
                                       for loc in spill_composition[key] if loc==b], \
                                      histtype='step', label=twoBytwo_defs.loc_dict[b], linewidth=2)
            ax[a[i][0]][a[i][1]].grid(True)
            ax[a[i][0]][a[i][1]].legend()
            ax[a[i][0]][a[i][1]].set_title(r'Fiducial Visible Energy $\nu$ Origin'+'\n'+'TPC '+str(i))
            ax[a[i][0]][a[i][1]].set_xlabel(r'Fraction of $\nu$ Interactions Per Spill')
            ax[a[i][0]][a[i][1]].set_ylabel('Spill Count')
    if save==True: plt.savefig('tpc_vertex_origin_composition.png')
    else: plt.show()
            


def plot_fv_spill_nu_count(d, save):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist([len(d[key]['vertex']) for key in d.keys()], bins=np.linspace(0,20,21), histtype='step', linewidth=2)
    ax.set_xlabel(r'$\nu$ Interactions per Spill')
    ax.set_ylabel('Spill Count')
    ax.set_yscale('log')
    ax.grid(True)
    if save==True: plt.savefig('fv_nu_count.png')
    else: plt.show()



def plot_tpc_spill_nu_count(d, save):
    fig, ax = plt.subplots(figsize=(6,4))
    for i in range(8):
        ax.hist([len(d[key]['vertex'][i]) for key in d.keys()], \
                bins=np.linspace(0,20,21), histtype='step', linewidth=2, label=str(i))
    ax.set_xlabel(r'$\nu$ Interactions per Spill')
    ax.set_ylabel('Spill Count')
    ax.set_yscale('log')
    ax.legend(title='TPC')
    ax.grid(True)
    if save==True: plt.savefig('tpc_nu_count.png')
    else: plt.show()
    


def plot_fv_contained_e(d, save):
    a={}
    for key in d.keys():
        a[key]=0.
        for i in range(len(d[key]['status'])):
            if d[key]['status'][i]=='tg': continue
            a[key]+=d[key]['contained_e'][i]
    bins=np.linspace(0,1000,41)            
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist([np.sum(d[key]['contained_e']) for key in d.keys()],\
             bins=bins, histtype='step', linewidth=2, label='all tracks')
    ax.hist([a[key] for key in a.keys()],\
             bins=bins, histtype='step', linewidth=2, label='partially- & full-contained tracks')
    ax.set_title('Fiducial Volume Contained Visible Energy')
    ax.set_xlabel('Visible Energy [MeV]')
    ax.set_ylabel('Spill Count / 25 MeV')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)
    if save==True: plt.savefig("fv_contained_e.png")
    else: plt.show()


def plot_tpc_contained_e(d, save):
    a={}
    for key in d.keys():
        a[key]=0.
        for i in range(8):
            for j in range(len(d[key]['status'][i])):
                if d[key]['status'][i][j]=='tg': continue
                a[key]+=d[key]['contained_e'][i][j]
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    bins=np.linspace(0,1000,41)
    for i in range(8):
        ax[0].hist([np.sum(d[key]['contained_e'][i]) for key in d.keys()],\
                   bins=bins, label=str(i), histtype='step')
        ax[1].hist([a[key] for key in a.keys()],bins=bins, histtype='step', label=str(i))
    for i in range(2):
        if i==0: ax[i].set_title('Fiducial Volume Contained Visible Energy'+'\n'+'All Tracks')
        if i==0: ax[i].set_title('Fiducial Volume Contained Visible Energy'+'\n'+'Partially- and Fully-contained Tracks')
        ax[i].set_xlabel('Visible Energy [MeV]')
        ax[i].set_ylabel('Spill Count / 25 MeV')
        ax[i].set_yscale('log')
        ax[i].legend(title='TPC')
        ax[i].grid(True)
    if save==True: plt.savefig("tpc_contained_e.png")
    else: plt.show()


    
def plot_fv_n_tracks(d, save):
    t,f,p=[{} for i in range(3)]
    for key in d.keys():
        t[key]=0; f[key]=0; p[key]=0
        for s in d[key]['status']:
            if s=='tg': t[key]+=1
            elif s=='fc': f[key]+=1
            else: p[key]+=1
    bins=np.linspace(0,200,41)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist([len(d[key]['pdg']) for key in d.keys()], bins=bins, \
            histtype='step', linewidth=2, label='all')
    ax.hist([t[key] for key in t.keys()], bins=bins, \
            histtype='step', linewidth=2, label='through-going')
    ax.hist([p[key] for key in p.keys()], bins=bins, \
            histtype='step', linewidth=2, label='partially-contained')
    ax.hist([f[key] for key in f.keys()], bins=bins, \
            histtype='step', linewidth=2, label='fully-contained')
    ax.set_title('Fiducialized Tracks'+'\n'+'(5 MeV fiducial volume threshold)')
    ax.set_xlabel('Visible Particles Per Spill [Count]')
    ax.set_ylabel('Spill Count')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True)
    if save==True: plt.savefig('fv_tracks_count.png')
    else: plt.show()



def plot_tpc_n_tracks(d, save):
    fig, ax = plt.subplots(2,2,figsize=(12,12))
    t,f,p=[{} for i in range(3)]
    for key in d.keys():
        t[key]={}; f[key]={}; p[key]={}
        for i in range(8):
            t[key][i]=0; f[key][i]=0; p[key][i]=0
            for s in d[key]['status'][i]:
                if s=='tg': t[key][i]+=1
                elif s=='fc': f[key][i]+=1
                else: p[key][i]+=1
    bins=np.linspace(0,200,41)
    for i in range(8):
        ax[0][0].hist([len(d[key]['pdg'][i]) for key in d.keys()], \
                      bins=bins, histtype='step', linewidth=2, label=str(i))
        ax[0][1].hist([t[key][i] for key in t.keys() for i in range(8)], \
                      bins=bins, histtype='step', linewidth=2, label=str(i))
        ax[1][0].hist([p[key][i] for key in p.keys() for i in range(8)], \
                      bins=bins, histtype='step', linewidth=2, label=str(i))
        ax[1][1].hist([f[key][i] for key in f.keys() for i in range(8)], \
                      bins=bins, histtype='step', linewidth=2, label=str(i))
    for i in range(2):
        for j in range(2):
            if i==0 and j==0: ax[i][j].set_title('All Tracks')
            if i==0 and j==1: ax[i][j].set_title('Through-going Tracks')
            if i==1 and j==0: ax[i][j].set_title('Partially-contained Tracks')
            if i==1 and j==1: ax[i][j].set_title('Fully-contained Tracks')
            ax[i][j].set_xlabel('Visible Particles Per Spill [Count]')
            ax[i][j].set_ylabel('Spill Count')
            ax[i][j].set_yscale('log')
            ax[i][j].legend(title='TPC')
            ax[i][j].grid(True)
    if save==True: plt.savefig('tpc_tracks_count.png')
    else: plt.show()


    
def main(sim_file, save):
    fv_dict=dict(); tpc_dict=dict()

    file_ctr=0
    for sim_file in glob.glob(file_dir+'*.EDEPSIM.h5'):
        file_ctr+=1
        if file_ctr>100: break
        print('FILE #: ',file_ctr)
        sim_h5 = h5py.File(sim_file,'r')

        unique_spill = np.unique(sim_h5['trajectories']['spillID'])
        for spill_id in unique_spill:
            ghdr, gstack, traj, vert, seg = auxiliary.get_spill_data(sim_h5, spill_id)
            fiducial_volume_multiplicity(spill_id, seg, vert, traj, fv_dict)
            tpc_multiplicity(spill_id, seg, vert, traj, tpc_dict)

    auxiliary.save_dict_to_json(fv_dict, 'fv_multiplicity', False)
#    plot_fv_mult(fv_dict, save=save)

    auxiliary.save_dict_to_json(tpc_dict, 'tpc_multiplicity', False)
#    plot_tpc_mult(tpc_dict, save=save)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--sim_file', default=None, required=False, type=str, help='''string corresponding to the path of the edep-sim ouput simulation file to be considered''')
    parser.add_argument('-s', '--save', default=True, type=bool, help='''Save plot to PNG if true; otherwise, show in screen''')
    args = parser.parse_args()
    main(**vars(args))
