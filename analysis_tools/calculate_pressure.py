from copy import deepcopy
from numpy import power, pi, rint, sqrt, median, log, mean, percentile, sum, array
from numpy.random import choice, seed
from heapq import heappush, heappushpop, heapify, heappop

#calculates the pressure based on entropy loss estimates
def CalculatePressure(frames):
    etas = []
    max_etas = []
    seed(seed=1)
    
    for frame in frames:
        #various local copies
        coords = deepcopy(frame['coords'])
        diameters = frame['diameters']
        diameters_scaled = frame['diameters']/frame['diameters'][0]
        frame_features = []
        combined_data = zip(coords, diameters)
        
        #current volume fraction and number of particles
        eta = (pi/6.0)*sum(power(diameters, 3.0))/power(frame['L'], 3.0)
        N = len(diameters)

        #find closest particles
        smallest_scale = 1000000000000.0
        for particle, diameter in combined_data:
            #nearest neighbor coordinate wrapping
            Rpj = particle - coords
            Rpj = Rpj - rint(Rpj/frame['L'])*frame['L']
            Rpj = (sqrt(sum(power(Rpj, 2.0), axis=1)))

            #generate statistics for various nearest neighbors
            #sorter = Rpj.argsort()
            #Rpj = Rpj[sorter[::1]]
            #Dpj = diameters[sorter[::1]]

            #cross particle diameters
            #Dpj = (1.0/2.0)*(diameter + Dpj)
            Dpj = (1.0/2.0)*(diameter + diameters)
            
            #compute pair scales and find the closest
            scales = Rpj/Dpj
            
            #find the smallest two scales with O(N): the first is the self overlap and the second is the real smallest
            smallest_two_scales = [-1000000000000.0, -1000000000000.0]
            heapify(smallest_two_scales)
            for scale in scales:
                heappushpop(smallest_two_scales, -scale)
            second_smallest_scale = -heappop(smallest_two_scales)
            smallest_scale = min(second_smallest_scale, smallest_scale)
            
            #smallest_scale = min(min(scales[1:]), smallest_scale)
        
        #calculate the maximum eta possible
        max_eta = power(smallest_scale, 3.0)*eta
    
        etas.append(eta)
        max_etas.append(max_eta)
        #max_etas.append(smallest_scale)
    max_etas=array(sorted(max_etas))
    
    #bootstrap the max_etas to get error bars and an average pressure
    pressures = []
    num_frames = len(max_etas)
    for it in range(10000):
        max_etas_btsp = choice(max_etas, num_frames)
        eta_new = median(max_etas_btsp)
        frac = float(sum(max_etas_btsp >= eta_new))/float(num_frames)
        d_eta = eta_new - eta
        P = -eta*(log(frac)/float(N))/d_eta
        pressures.append(P)
    return (mean(pressures), percentile(pressures, 5), percentile(pressures, 95))