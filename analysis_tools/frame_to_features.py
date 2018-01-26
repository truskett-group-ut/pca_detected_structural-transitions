from copy import deepcopy
from numpy import power, rint, sqrt, sum

#this generates NN features for the PCA analysis (or some other machine learning method as well)
def FrameToFeatures(frame, N_nn, method, step):
    #work with dimensionless coordinates so density is not explicitly buried
    coords = deepcopy(frame['coords'])#/frame['L']
    
    N = float(len(coords))
    V = power(frame['L'], 3.0)
    normalizing_distance = power(V/N, 1.0/3.0)
    
    diameters = frame['diameters']
    diameters_scaled = frame['diameters']/frame['diameters'][0]
    
    frame_features = []
    combined_data = zip(coords, diameters)
    for particle, diameter in combined_data[0::step]:
        #nearest neighbor coordinate wrapping
        Rpj = particle - coords
        Rpj = Rpj - rint(Rpj/frame['L'])*frame['L']
        Rpj = (sqrt(sum(power(Rpj, 2.0), axis=1)))     
        
        #generate statistics for various nearest neighbors
        sorter = Rpj.argsort()
        Rpj = Rpj[sorter[::1]]
        Dpj = diameters_scaled[sorter[::1]]
        
        #possible feature options
        if method == 'distance':
            frame_features.extend((Rpj[1:N_nn+1]/normalizing_distance))
        elif method == 'composition':
            frame_features.extend(Dpj[1:N_nn+1])

    return list(frame_features)