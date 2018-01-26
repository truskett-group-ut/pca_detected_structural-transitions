from copy import deepcopy
from numpy import zeros, mean, nanmean, histogram, inf
from asap3.analysis import PTM
from asap3 import Atoms, EMT, units


#interface to the asap library for polyhedral template matching
def PolyTempMatch(frames, rmsd_max, bins=None):
    #initialize data storage
    num_frames = float(len(frames))
    structure = [[],[],[],[],[],[]] 
    rmsd = [[],[],[],[],[],[]] 
    if bins is not None:
        num_bins = len(bins) - 1
    else:
        num_bins = 0
    rmsd_hist = [zeros(num_bins), zeros(num_bins), zeros(num_bins), zeros(num_bins), zeros(num_bins), zeros(num_bins)]
    
    #load into ase format one frame at a time and analyze
    for frame in frames:
        L, coords = frame['L'], deepcopy(frame['coords'])
        N = len(coords)
        id_ = 'Ni{}'.format(N)
        atoms = Atoms(None, coords, numbers=len(coords)*[7], cell=[(L,0,0), (0,L,0), (0,0,L)], pbc=True)
        ptm = PTM(atoms, cutoff=L/2.0, rmsd_max=rmsd_max, 
                  target_structures=None, calculate_strains=False, 
                  quick=False, return_nblist=False)
        structure_single = deepcopy(ptm['structure'])
        structure_single.astype(int)
        rmsd_single = deepcopy(ptm['rmsd'])
        rmsd_single[rmsd_single == inf] = 0.0
        
        #bin the fraction according to type
        structure[0].append(mean(structure_single == 0)) #none
        structure[1].append(mean(structure_single == 1)) #fcc
        structure[2].append(mean(structure_single == 2)) #hcp
        structure[3].append(mean(structure_single == 3)) #bcc
        structure[4].append(mean(structure_single == 4)) #ico
        structure[5].append(mean(structure_single == 5)) #sc
        
        #bin the rmsd according to type
        rmsd[0].append(mean(rmsd_single[structure_single == 0])) #none
        rmsd[1].append(mean(rmsd_single[structure_single == 1])) #fcc
        rmsd[2].append(mean(rmsd_single[structure_single == 2])) #hcp
        rmsd[3].append(mean(rmsd_single[structure_single == 3])) #bcc
        rmsd[4].append(mean(rmsd_single[structure_single == 4])) #ico
        rmsd[5].append(mean(rmsd_single[structure_single == 5])) #sc
        
        #bin all the rmsd data to create a histogram with
        if bins is not None:
            rmsd_hist[0] = rmsd_hist[0] + histogram(rmsd_single[structure_single == 0], bins=bins)[0]
            rmsd_hist[1] = rmsd_hist[1] + histogram(rmsd_single[structure_single == 1], bins=bins)[0]
            rmsd_hist[2] = rmsd_hist[2] + histogram(rmsd_single[structure_single == 2], bins=bins)[0]
            rmsd_hist[3] = rmsd_hist[3] + histogram(rmsd_single[structure_single == 3], bins=bins)[0]
            rmsd_hist[4] = rmsd_hist[4] + histogram(rmsd_single[structure_single == 4], bins=bins)[0]
            rmsd_hist[5] = rmsd_hist[5] + histogram(rmsd_single[structure_single == 5], bins=bins)[0]
        
    #perform final averages
    structure_mean = [nanmean(structure[0]), nanmean(structure[1]), nanmean(structure[2]), 
                     nanmean(structure[3]), nanmean(structure[4]), nanmean(structure[5])]  
    rmsd_mean = [nanmean(rmsd[0]), nanmean(rmsd[1]), nanmean(rmsd[2]), 
                nanmean(rmsd[3]), nanmean(rmsd[4]), nanmean(rmsd[5])]
    
    return (structure_mean, rmsd_mean, rmsd_hist, structure)