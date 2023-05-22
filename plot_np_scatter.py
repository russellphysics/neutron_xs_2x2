import matplotlib
import matplotlib.pyplot as plt
import h5py
import argparse
import numpy as np
import json
import twoBytwo_defs
import auxiliary

pdg_label={211:r'$\pi^+$',-211:r'$\pi^-$',13:'$\mu^-$',-13:'$\mu^+$',2212:'p'}

#### signal = elastic scatter fiducialized proton via neutron and fiducialized nu vertex

def plot_single_track_pid(d):
    fig, ax = plt.subplots(figsize=(8,6))
    particle=[d[key] for key in d.keys()]
    particle_set=set(particle)
    particle_count=[(p, particle.count(p)) for p in particle_set]
    particle_fraction=[100*(i[1]/len(particle))for i in particle_count]
    particle_label=[pdg_label[i[0]] for i in particle_count]
    ax.pie(particle_fraction, labels=particle_label, autopct='%1.1f%%')
    ax.set_title('n-p Inelastic Scattering'+'\n'+r'with Single Charged Track at $\nu$ Vertex')
    print(particle_count)
    plt.show()



def plot_multiplicity(d):
    fig, ax = plt.subplots(1,3,figsize=(18,6))

    neutron=[d[key]['neutron'] for key in d.keys()]
    neutron_set=set(neutron)
    neutron_count=[(p, neutron.count(p)) for p in neutron_set]
    neutron_fraction=[100*(i[1]/len(neutron))for i in neutron_count]
    neutron_label=[str(i[0]) for i in neutron_count]
    ax[0].pie(neutron_fraction, labels=neutron_label, autopct='%1.1f%%')
    ax[0].set_title('Primary Neutron Multiplicity')

    shower=[d[key]['shower'] for key in d.keys()]
    shower_set=set(shower)
    shower_count=[(p, shower.count(p)) for p in shower_set]
    shower_fraction=[100*(i[1]/len(shower))for i in shower_count]
    shower_label=[str(i[0]) for i in shower_count]
    ax[1].pie(shower_fraction, labels=shower_label, autopct='%1.1f%%')
    ax[1].set_title('Primary Electromagnetic Multiplicity')

    track=[d[key]['track'] for key in d.keys()]
    track_set=set(track)
    track_count=[(p, track.count(p)) for p in track_set]
    track_fraction=[100*(i[1]/len(track))for i in track_count]
    track_label=[str(i[0]) for i in track_count]
    ax[2].pie(track_fraction, labels=track_label, autopct='%1.1f%%')
    ax[2].set_title('Primary Charged Track Multiplicity')

    plt.show()






z={'s':'primary n progenitor'+'\n'+r'with fiducial $\nu$ vertex',
   'a':r'fiducial $\nu$ progenitor'+'\n'+'with charged'+'\n'+'track primary'+'\n'+'final states',
   'b':r'fiducial $\nu$ progenitor'+'\n'+'without charged'+'\n'+'track primary'+'\n'+'final states',
   'c':r'primary charged track progenitor'+'\n'+r'with fiducial $\nu$ vertex',
   'd':r'$\nu$ vertex outside FV'+'\n'+'and another $\nu$ vertex in FV',
   'e':'primary n progenitor'+'\n'+r'with no $\nu$ vertex in FV'}

def plot_parent_pdg(d, save):
    z={2112:'n',2212:'p',211:r'$\pi^+$',-211:r'$\pi^-$',-14:r'$\overline{\nu}_\mu$',14:r'$\nu_\mu$',
       -12:r'$\overline{\nu_e}$',12:r'$\nu_e$',22:r'$\gamma$',3122:r'$\Lambda$'}
    pdg=[d[key]['parent_pdg'] for key in d.keys()]
    pdg_set=set(pdg)
    pdg_count=[(p, pdg.count(p)) for p in pdg_set]
    pdg_fraction=[100*(i[1]/len(pdg)) for i in pdg_count]
    pdg_label=[]
    for i in pdg_count:
        if i[0] in z.keys(): pdg_label.append(z[i[0]])
        else: pdg_label.append(str(i[0]))
    fig, ax = plt.subplots(figsize=(8,6))
    ax.pie(pdg_fraction, labels=pdg_label, autopct='%1.1f%%')
    ax.set_title('Proton Parent PDG Code')
    #if save==True: plt.savefig('proton_parent_pdg.png')
    #else: plt.show()
    plt.show()


def plot_nu_location(d, save):
    loc=[d[key]['vertex'] for key in d.keys()]
    loc_set=set(loc)
    loc_count=[(p, loc.count(p)) for p in loc_set]
    loc_fraction=[100*(i[1]/len(loc)) for i in loc_count]
    loc_label=[twoBytwo_defs.loc_dict[str(i[0])] for i in loc_count]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.pie(loc_fraction, labels=loc_label, autopct='%1.1f%%')
    ax.set_title(r'$\nu$ Vertex Location')
    
    if save==True: plt.savefig('proton_nu_location.png')
    else: plt.show()


    
def plot_len_vs_dis(d, save):
    lbins=np.linspace(0,20,51)
    dbins=np.linspace(0,400,51)
    n_pdg=[2112]
    nu_pdg=[12,-12,14,-14,15,-15]
    charged_pion_pdg=[211,-211]
    p_pdg=[2212]
    labels={'n':n_pdg, r'$\nu$':nu_pdg, r'$\pi^{+/-}$':charged_pion_pdg,
            'p':p_pdg}
    fig, ax = plt.subplots(2,2,figsize=(12,12))
    ctr=0
    c={0:(0,0),1:(0,1),2:(1,0),3:(1,1)}
    for k,v in labels.items():
        ax[c[ctr][0]][c[ctr][1]].hist2d([d[key]['p_len'] for key in d.keys() \
                                         if d[key]['parent_pdg'] in v and d[key]['p_len']<=20. and
                                         d[key]['nu_p_d']<=400.], \
                                        [d[key]['nu_p_d'] for key in d.keys() \
                                         if d[key]['parent_pdg'] in v and d[key]['p_len']<=20. and
                                         d[key]['nu_p_d']<=400.], \
                                        bins=[lbins,dbins], norm=matplotlib.colors.LogNorm())
        ax[c[ctr][0]][c[ctr][1]].set_xlabel('Proton Track Length [cm]')
        ax[c[ctr][0]][c[ctr][1]].set_ylabel(r'$\nu$ vertex to proton start point [cm]')
        ax[c[ctr][0]][c[ctr][1]].set_title(k+' Parent')
        ax[c[ctr][0]][c[ctr][1]].grid(True)
        ctr+=1
    if save==True: plt.savefig('fv_proton_length_vs_distance.png')
    else: plt.show()



def plot_len_vs_e(d, save):
    lbins=np.linspace(0,100,51)
    dbins=np.linspace(0,500,51)
    n_pdg=[2112]
    nu_pdg=[12,-12,14,-14,15,-15]
    charged_pion_pdg=[211,-211]
    p_pdg=[2212]
    labels={'n':n_pdg, r'$\nu$':nu_pdg, r'$\pi^{+/-}$':charged_pion_pdg,
            'p':p_pdg}
    fig, ax = plt.subplots(2,2,figsize=(12,12))
    ctr=0
    c={0:(0,0),1:(0,1),2:(1,0),3:(1,1)}
    for k,v in labels.items():
        ax[c[ctr][0]][c[ctr][1]].hist2d([d[key]['p_len'] for key in d.keys() \
                                         if d[key]['parent_pdg'] in v and d[key]['p_len']<=100. and
                                         d[key]['p_vis_e']<=500.], \
                                        [d[key]['p_vis_e'] for key in d.keys() \
                                         if d[key]['parent_pdg'] in v and d[key]['p_len']<=100. and
                                         d[key]['p_vis_e']<=500.], \
                                      bins=[lbins,dbins], norm=matplotlib.colors.LogNorm())
        ax[c[ctr][0]][c[ctr][1]].set_xlabel('Proton Track Length [cm]')
        ax[c[ctr][0]][c[ctr][1]].set_ylabel(r'Proton Visible Energy [MeV]')
        ax[c[ctr][0]][c[ctr][1]].set_title(k+' Parent')
        ax[c[ctr][0]][c[ctr][1]].grid(True)
        ctr+=1
    if save==True: plt.savefig('fv_proton_length_vs_energy.png')
    else: plt.show()
    
    
    
def plot_p_length(d, save):
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    sbins=np.linspace(0,10,51)
    lbins=np.linspace(0,100,51)
    other=[]
    n_pdg=[2112]; other+=n_pdg
    nu_pdg=[12,-12,14,-14,15,-15]; other+=nu_pdg
    charged_pion_pdg=[211,-211]; other+=charged_pion_pdg
    proton_pdg=[2212]; other+=proton_pdg
    labels={'n':n_pdg, r'$\nu$':nu_pdg, r'$\pi^{+/-}$':charged_pion_pdg,
            'p':proton_pdg, 'other':other}
    for k,v in labels.items():
        if k=='other': continue
        ax[0].hist([d[key]['p_len'] for key in d.keys() \
                    if d[key]['parent_pdg'] in v and d[key]['p_len']<=100.], \
                   bins=lbins, histtype='step',linewidth=2,label=k)
        ax[1].hist([d[key]['p_len'] for key in d.keys() \
                    if d[key]['parent_pdg'] in v and d[key]['p_len']<=10.],
                   bins=sbins, \
                   histtype='step',linewidth=2,label=k)
    ax[0].hist([d[key]['p_len'] for key in d.keys() \
                if d[key]['parent_pdg'] not in other and d[key]['p_len']<=100.], \
               bins=lbins, histtype='step',linewidth=2,label='other')
    ax[1].hist([d[key]['p_len'] for key in d.keys() \
                if d[key]['parent_pdg'] not in other and d[key]['p_len']<=10.],
               bins=sbins, \
               histtype='step',linewidth=2,label='other')
    for i in range(2):
        ax[i].set_xlabel('Proton Track Length [cm]')
        ax[i].set_ylabel('Count')
        ax[i].grid(True)
        ax[i].set_yscale('log')
        if i==0: ax[i].legend(title='Proton Parent')
#    if save==True: plt.savefig('fv_proton_length.png')
#    else: plt.show()
    plt.show()


#def keyA_to_valueB(key, fv_nu):
#    spill_id = key.split('-')[0]
#    vertex_id = key.split('-')[1]
#    track_id = key.split('-')[-1]
#    if spill_id in fv_nu.keys():
#        if vertex_id in fv_nu[spill_id]:
#            if track_id in fv_nu[spill_id][vertex_id]:


def plot_p_length_class(d, c):
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    sbins=np.linspace(0,10,51)
    lbins=np.linspace(0,100,51)
    print(c.keys)

    x={}
    for key in d.keys():
        if key not in c.keys(): continue
        if key not in x: x[key]={}
        if c[key] not in x[key]: x[key][c[key]]=d[key]['p_len']
    
    for k,v in z.items():
        
        ax[0].hist([x[keyA][keyB] for keyA in x.keys() for keyB in x[keyA] \
                    if keyB==k],
                   bins=lbins, histtype='step', label=v)
        ax[1].hist([x[keyA][keyB] for keyA in x.keys() for keyB in x[keyA] \
                    if keyB==k],
                   bins=sbins, histtype='step', label=v)

    for i in range(2):
        ax[i].set_xlabel('Proton Track Length [cm]')
        ax[i].set_ylabel('Count')
        ax[i].grid(True)
        ax[i].set_yscale('log')
        if i==0: ax[i].legend(title='Proton Parent')
    plt.show()
    
    
    
def plot_nu_p_dis(d, save):
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    sbins=np.linspace(0,50,51)
    lbins=np.linspace(0,1000,51)
    other=[]
    n_pdg=[2112]; other+=n_pdg
    nu_pdg=[12,-12,14,-14,15,-15]; other+=nu_pdg
    charged_pion_pdg=[211,-211]; other+=charged_pion_pdg
    proton_pdg=[2212]; other+=proton_pdg
    labels={'n':n_pdg, r'$\nu$':nu_pdg, r'$\pi^{+/-}$':charged_pion_pdg,
            'p':proton_pdg, 'other':other}
    for k,v in labels.items():
        if k=='other': continue
        ax[0].hist([d[key]['nu_p_d'] for key in d.keys() \
                    if d[key]['parent_pdg'] in v],bins=lbins, \
                   histtype='step',linewidth=2,label=k)
        ax[1].hist([d[key]['nu_p_d'] for key in d.keys() \
                    if d[key]['parent_pdg'] in v and d[key]['nu_p_d']<=50.],
                   bins=sbins, \
                   histtype='step',linewidth=2,label=k)
    ax[0].hist([d[key]['nu_p_d'] for key in d.keys() \
                if d[key]['parent_pdg'] not in other],bins=lbins, \
               histtype='step',linewidth=2,label='other')
    ax[1].hist([d[key]['nu_p_d'] for key in d.keys() \
                if d[key]['parent_pdg'] not in other and d[key]['nu_p_d']<=50.],
               bins=sbins, \
               histtype='step',linewidth=2,label='other')
    for i in range(2):
        ax[i].set_xlabel(r'$\nu$ vertex to proton start point [cm]')
        ax[i].set_ylabel('Count')
        ax[i].set_yscale('log')
        ax[i].grid(True)
        if i==0: ax[i].legend(title='Proton Parent')
    if save==True: plt.savefig('nu_p_distance.png')
    else: plt.show()



def plot_nu_p_dt(d, save):
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    sbins=np.linspace(0,0.1,51)
    lbins=np.linspace(0,2,51)
    other=[]
    n_pdg=[2112]; other+=n_pdg
    nu_pdg=[12,-12,14,-14,15,-15]; other+=nu_pdg
    charged_pion_pdg=[211,-211]; other+=charged_pion_pdg
    proton_pdg=[2212]; other+=proton_pdg
    labels={'n':n_pdg, r'$\nu$':nu_pdg, r'$\pi^{+/-}$':charged_pion_pdg,
            'p':proton_pdg, 'other':other}
    for k,v in labels.items():
        if k=='other': continue
        ax[0].hist([d[key]['nu_p_dt'] for key in d.keys() \
                    if d[key]['parent_pdg'] in v],bins=lbins, \
                   histtype='step',linewidth=2,label=k)
        ax[1].hist([d[key]['nu_p_dt'] for key in d.keys() \
                    if d[key]['parent_pdg'] in v and d[key]['nu_p_dt']<=50.],
                   bins=sbins, \
                   histtype='step',linewidth=2,label=k)
    ax[0].hist([d[key]['nu_p_dt'] for key in d.keys() \
                if d[key]['parent_pdg'] not in other],bins=lbins, \
               histtype='step',linewidth=2,label='other')
    ax[1].hist([d[key]['nu_p_dt'] for key in d.keys() \
                if d[key]['parent_pdg'] not in other and d[key]['nu_p_dt']<=50.],
               bins=sbins, \
               histtype='step',linewidth=2,label='other')
    for i in range(2):
        ax[i].set_xlabel(r'$\Delta$t$_{p\nu}$ [$\mu$s]')
        ax[i].set_ylabel('Count')
        ax[i].set_yscale('log')
        ax[i].grid(True)
        if i==0: ax[i].legend(title='Proton Parent')
    if save==True: plt.savefig('nu_p_dt.png')
    else: plt.show()


    
def plot_spill_mult(d, save):
    spill_mult={}
    for key in d.keys():
        if key[0] not in spill_mult: spill_mult[key[0]]=0
        spill_mult[key[0]]+=1
    fig, ax = plt.subplots(figsize=(6,4))
    bins=np.linspace(0,20,21)
    ax.hist([spill_mult[key] for key in spill_mult.keys()], \
            bins=bins,histtype='step',linewidth=2, color='b')
    axtwin=ax.twinx()
    axtwin.hist([spill_mult[key] for key in spill_mult.keys()], \
                bins=bins,histtype='step',linewidth=2,linestyle='dashed',\
                cumulative=True, density=True, color='r')
    ax.set_xlabel('Fiducialized Proton Spill Multiplicity')
    ax.set_ylabel('Spill Count')
    axtwin.set_ylabel('Cumulative Probability')
    ax.grid(True)
    if save==True: plt.savefig('fv_proton_mult.png')
    else: plt.show()



def breakup_dict_key(d):
    b={}
    for key in d.keys():
        spill = int(key.split("-")[0])
        vertex = int(key.split("-")[1])
        track_id = int(key.split("-")[-1])
        if spill not in b.keys(): b[spill]={}
        if vertex not in b[spill]: b[spill][vertex]={}
        if track_id not in b[spill][vertex]: b[spill][vertex][track_id]=0
        b[spill][vertex][track_id]+=1
    return b
        
    

def classify_events(proton, fv_primary_nu_multiplicity):
    in_tpc=[]
    for i in range(8): in_tpc.append(str(i))
    output={}; signal_particle_multiplicity=dict(); signal_pid={}
    for key in proton.keys():
        spill_id = key.split("-")[0]
        vertex_id = key.split("-")[1]
        track_id = key.split("-")[-1]
        
        parent_pdg = proton[key]['parent_pdg']
        if proton[key]['vertex'] in in_tpc: # fiducial nu
            if parent_pdg==2112:
                output[key]='s' # signal; neutron parent
                neutron_primaries=0; shower_primaries=0; track_primaries=0
                primaries=proton[key]['primaries']
                track_pdg=-1
                for p in primaries:
                    if p==2112: neutron_primaries+=1
                    elif p==22 or p==-11 or p==11 or p==111: shower_primaries+=1
                    elif p==13 or p==-13 or p==211 or p==-211 or p==2212 or p==-2212: track_primaries+=1; track_pdg=p
                signal_particle_multiplicity[key]=dict(
                    neutron=neutron_primaries,
                    shower=shower_primaries,
                    track=track_primaries)
                if track_primaries==1: signal_pid[key]=track_pdg
                    
            else:
                if parent_pdg in [12, 14, 16, -12, -14, -16]: # nu parent
                    
                    neutron_primaries=0
                    primaries=proton[key]['primaries']
                    for p in primaries:
                        if p==2112: neutron_primaries+=1
                    charged_primaries=len(primaries)-neutron_primaries

                    if charged_primaries>0: output[key]='a' # 'tag-able' charged tracks
                    else: output[key]='b' # 'irreducible' single visible final state (proton)
                else: # non-nu and non-neutron parent
                    output[key]='c' #'tag-able' charged parent track
        else: # non-fiducial nu
            if parent_pdg==2112:
                ### ASK if key in fv_nu*.json dictionary
                if spill_id in fv_primary_nu_multiplicity.keys():
                    # if True: vertex not in active volume & another vertex in active volume --> may be tagable by timing
                    output[key]='d'
                else:
                    # if False: vertex not in active volume & no vertex in active volume
                    output[key]='e' # untag-able vertex --> how often another vertex in active volume
    return output, signal_particle_multiplicity, signal_pid


def plot_signal_gap(proton, c, signal_pid):
    l,d = [[] for i in range(2)]
    all_signal=0
    survived_signal=0

    len_vertex_cut={}
    distance = [0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0]
    for i in distance:
        for j in distance: len_vertex_cut[(i,j)]=0
    
    for key in proton.keys():
        if key not in c.keys(): continue
        if c[key]!='s': continue
        if key not in signal_pid.keys(): continue
        if abs(signal_pid[key])!=13: continue
        proton_length=proton[key]['p_len']
        nu_p_d=proton[key]['nu_p_d']
        l.append(proton_length)
        d.append(nu_p_d)
        for akey in len_vertex_cut.keys():
            if akey[0]>=proton_length and akey[1]>=nu_p_d:
                len_vertex_cut[akey]+=1
        all_signal+=1
        if proton[key]['p_len']>1.25 and proton[key]['nu_p_d']>2.5:
            survived_signal+=1

    print('All signal: ',all_signal)
    print('Signal after cuts: ',survived_signal)

    print('100 of 1023 files analyzed')
    print('-------------------------')
    print('scaling to 2.5E19 POT RHC')
    scaling_factor=(2.5e19)/((100*10e19)/1023)
    print('All signal (scaled): ',all_signal*scaling_factor)
    print('Signal after cuts: ',survived_signal*scaling_factor)
        
    a=[]
    for i in distance:
        temp=[]
        for j in distance: temp.append(len_vertex_cut[(i,j)]*scaling_factor)
        a.append(temp)

    fig0, ax0 = plt.subplots(figsize=(8,8))
    CS=ax0.contour(a, levels=7)
    ax0.clabel(CS, inline=True, fontsize=10)
    ax0.set_ylabel(r'$\nu vertex to proton start distance [cm]$')
    ax0.set_xlabel(r'proton track length [cm]')
    plt.show()
            
    bins=np.linspace(0,100,41)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.hist2d(l,d,bins=[np.linspace(0,50,41),bins], norm=matplotlib.colors.LogNorm())
    ax.set_xlabel('Proton Length [cm]')
    ax.set_ylabel(r'$\nu$ Vertex to Proton Start Sistance [cm]')
    ax.axvline(x=1.25, color='r', linestyle='dashed', label='1.25 cm proton track length threshold')
    ax.axhline(y=2.5, color='orange', linestyle='dashed', label='2.5 cm vertex gap threshold')
    ax.grid(True)
    ax.legend()
    plt.show()
    
    

def plot_event_classification(c):
    a=[c[key] for key in c.keys()]
    a_set=set(a)
    a_count=[(p, a.count(p)) for p in a_set]
    a_fraction=[100*(i[1]/len(a)) for i in a_count]
    a_label=[]
    for i in a_count:
        if i[0] in z.keys(): a_label.append(z[i[0]])
        else: a_label.append(str(i[0]))
    fig, ax = plt.subplots(figsize=(6,4))
    ax.pie(a_fraction, labels=a_label, autopct='%1.1f%%')
    ax.set_title('Active Volume Fully-Contained Protons')
    plt.show()
            

def main(save):
    with open('neutron_tof_100files.json') as input_file:
        proton = json.load(input_file)

    with open('fv_nu_100files.json') as input_file:
        fv_nu_primaries = json.load(input_file)
    fv_proton_nu_multiplicity = breakup_dict_key(proton)
    fv_primary_nu_multiplicity = breakup_dict_key(fv_nu_primaries)
    c, signal_particle_multiplicity, signal_pid = classify_events(proton, fv_primary_nu_multiplicity)

#    plot_event_classification(c)
    plot_signal_gap(proton, c, signal_pid)
#    plot_multiplicity(signal_particle_multiplicity)
    #plot_single_track_pid(signal_pid)
#    plot_parent_pdg(proton, save)
#    plot_p_length_class(proton, c)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--save',default=True, type=bool, \
                        help='''save plot(s) to PNG; otherwise, show in screen ''')
    args = parser.parse_args()
    main(**vars(args))
