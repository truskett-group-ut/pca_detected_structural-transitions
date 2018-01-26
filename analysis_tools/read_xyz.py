import re
from numpy import array, power, std, mean, unique, concatenate
from numpy.random import shuffle
from copy import deepcopy

def ReadXYZ(filename, N, N_types, shuffle_data=True):
    #regex used to parse the data
    xyz_regex = r'(?:([a-zA-Z]+)\s+([0-9\.e\-\+]+)\s+([0-9\.e\-\+]+)\s+([0-9\.e\-\+]+)\s+([0-9\.e\-\+]+))'
    L_regex = r'(?:L\s*=\s*([0-9\.e\-\+]+))'
    
    #read in the data and check for volume fraction consistency
    eta_stats = []
    L_stats = []
    with open(filename, "r") as ins:
        frames = [] #xyz and diameter
        coords, coords_count = [], 0
        diameters = []
        types = []
        L = None
        for line in ins:
            search = re.search(xyz_regex, line)
            if search and coords_count < N:
                coords.append(array([float(search.group(2)), float(search.group(3)), float(search.group(4))]))
                diameters.append(2.0*float(search.group(5)))
                types.append(search.group(1))
                coords_count = coords_count + 1
            elif coords:
                coords = array(coords)
                diameters = array(diameters)
                types = array(types)
                sorter = diameters.argsort()
                coords = coords[sorter]
                diameters = diameters[sorter]
                types = types[sorter]
                frames.append({'coords': array(coords), 'diameters': array(diameters), 'types': array(types), 'L': L})
                eta_stats.append(sum(power(diameters, 3.0)))
                coords, coords_count = [], 0
                diameters = []
                types = []
                L = None
                search_L = re.search(L_regex, line)
                if search_L:
                    L = float(search_L.group(1))
                    L_stats.append(L)
            else:
                search_L = re.search(L_regex, line)
                if search_L:
                    L = float(search_L.group(1))
                    L_stats.append(L)
                    
        #append final frame
        coords = array(coords)
        diameters = array(diameters)
        types = array(types)
        sorter = diameters.argsort()
        coords = coords[sorter]
        diameters = diameters[sorter]
        types = types[sorter]
        frames.append({'coords': array(coords), 'diameters': array(diameters), 'types': array(types), 'L': L})
        eta_stats.append(sum(power(diameters, 3.0)))
        
    #check that the volume fraction is consistent accross all frames
    if (max(eta_stats)-min(eta_stats))/mean(eta_stats) > 0.0000000001:
        raise Exception('Particle sizes seem inconsistent accross frames!!!')
        
    #check that the box size is consistent accross all frames
    if (max(L_stats)-min(L_stats))/mean(L_stats) > 0.0000000001:
        raise Exception('Box size seem inconsistent accross frames!!!')
        
    #perform random shuffle of identical particles coordinates to help facilitate learning    
    if shuffle_data:
        shuffled_frames = []
        for frame in frames:
            #extract local copies for organizational convenience
            coords = frame['coords']
            diameters = frame['diameters']
            types = frame['types']
            L = frame['L']
            
            #prepare for shuffle
            coords_shuffled = None
            unique_types, start, count = unique(types, return_index=True, return_inverse=False, return_counts=True, axis=None)
            start__end = zip(start, start+count)
            #print types
            
            #check for errors
            if len(start__end) != len(unique_types):
                raise Exception('Bad data!!!')
            
            #do the shuffling
            for start, end in start__end:
                grouped = deepcopy(coords[start:end])
                shuffle(grouped)
                if coords_shuffled is not None:
                    coords_shuffled = concatenate((coords_shuffled, grouped), axis=0)
                else:
                    coords_shuffled = deepcopy(grouped)
            shuffled_frames.append({'coords': array(coords_shuffled), 'diameters': array(diameters), 'types': array(types), 'L': L})
        
        #set the data
        frames = shuffled_frames
        shuffled_frames = []
        
        
    #perform a check that everything was read in and/or processed correctly
    for frame in frames:
        if len(frame['coords']) != N or len(frame['diameters']) != N or frame['L'] is None or len(set(types)) != N_types:
            raise Exception('Bad data!!!')
        else:
            continue
    
    return frames