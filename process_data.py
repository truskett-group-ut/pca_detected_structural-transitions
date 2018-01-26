###############################BASIC REQUIREMENTS###################################

from numpy import arange, mean, array
import re
import pickle
from sklearn.decomposition import IncrementalPCA

from analysis_tools.read_xyz import ReadXYZ
from analysis_tools.down_sample_frames import DownSampleFrames
from analysis_tools.frame_to_features import FrameToFeatures
from analysis_tools.calculate_pressure import CalculatePressure
from analysis_tools.poly_temp_match import PolyTempMatch

def Z(eta, s1, s2, s3):
    eta = array(eta)
    return 1.0/(1.0-eta) + (3.0*s1*s2/s3)*(eta/power((1.0-eta), 2.0)) + (s2**3.0/s3**2.0)*(((3.0-eta)*eta**2.0)/power((1.0-eta), 3.0))

def TrajectoryToFeatures(frames, N_nn, method, step):
    #print filename
    features = []
    for frame in frames:        
        features.append(FrameToFeatures(frame, N_nn, method, step))
    return features


##############################DATA ANALYSIS SPECS###################################

#define the data to analyze
base = '../300p_sims/inc_mono/'
location_data = [('{}step_{}/'.format(base, '{}'), (1, 50 + 1))]
N=300
N_types = 1
s1 = ?
s2 = ?
s3 = ?

#PCA related stuff
shuffle_data = True
incpca = IncrementalPCA(n_components=25, whiten=True)
step = 10
N_nn = 30

#PTM_related stuff
num_ptm_frames = 200
rmsd_max = 10.0

#initialize lists for everything
etas = []
Ps, Ps_05, Ps_95 = [], [], []
PMs, RMSDs = [], []
OPs = []

#############################DATA ANALYSIS PORTION###################################

#open txt files for writing out data as it comes in
Ps_file_stream = open('{}processed_data/{}'.format(base, 'Ps.txt'), 'a+')
Ps_file_stream.write('eta,Z,Zcs\n')
PMs_file_stream = open('{}processed_data/{}'.format(base,'PMs.txt'), 'a+')
PMs_file_stream.write('eta,F,FCC,HCP,BCC,ICO,SC\n')

#initial loop to do most everything
print 'Starting first phase ...\n'
for raw_filename, min_max in location_data:
    for i in range(min_max[0], min_max[1]):
        filename = raw_filename.format(i)
        
        #read etas
        with open((filename + 'extended_states.txt'), 'rb') as f:
            lines = ' '.join(f.readlines())
            eta = float(re.search(r'etas\s*=\s*\(([0-9\.]+)\)', lines).group(1))
            etas.append(eta)
            
        #read in frames and possibly shuffle for PCA
        frames = ReadXYZ((filename + 'trajectory.xyz'), N=N, N_types=N_types, shuffle_data=shuffle_data)
        
        #get a down sampled set of frames for PTM do resolve memory issues
        frames_ds = DownSampleFrames(frames, num_ptm_frames)
        
        #calculate pressures
        P, P_05, P_95 = CalculatePressure(frames)
        Ps.append(P)
        Ps_05.append(P_05)
        Ps_95.append(P_95)
        Ps_file_stream.write('{},{},{}\n'.format(eta,P,Z(eta,s1,s2,s3)))
        
        #perform PTM analysis
        pm, rmsd, _, _  = PolyTempMatch(frames_ds, rmsd_max=rmsd_max, bins=None)
        PMs.append(array(pm))
        RMSDs.append(array(rmsd))
        PMs_file_stream.write('{},{},{},{},{},{},{}\n'.format(eta,pm[0],pm[1],pm[2],pm[3],pm[4],pm[5]))
        
        #train PCA
        features = TrajectoryToFeatures(frames, N_nn=N_nn, method='distance', step=step)
        incpca.partial_fit(features)
        
        print 'Wrote pressure and PTM results for dataset {}\n'.format(i)

#close file streams        
Ps_file_stream.close()
PMs_file_stream.close()

#save the P data
print 'Saving the P data...\n'
with open('{}processed_data/Ps.pkl'.format(base), 'wb') as file:
    pickle.dump(Ps, file)
    
#save the PM data
print 'Saving the PM data...\n'
with open('{}processed_data/PMs.pkl'.format(base), 'wb') as file:
    pickle.dump(PMs, file)

#save the PCA model
print 'Saving the PCA model...\n'
with open('{}processed_data/incpca.pkl'.format(base), 'wb') as file:
    pickle.dump(incpca, file)
    
    
    
##################################################################################################    
    
    
    
#open text file for PCA data    
OPs_file_stream = open('{}processed_data/{}'.format(base,'OPs.txt'), 'a+')   
OPs_file_stream.write('eta,P1,P2,P3,P4,P5,P6\n')

etas = []
#second loop to perform PCA analysis            
print 'Starting second phase ...\n'    
for raw_filename, min_max in location_data:
    for i in range(min_max[0], min_max[1]):
        filename = raw_filename.format(i)
        
        #read etas
        with open((filename + 'extended_states.txt'), 'rb') as f:
            lines = ' '.join(f.readlines())
            eta = float(re.search(r'etas\s*=\s*\(([0-9\.]+)\)', lines).group(1))
            etas.append(eta)
        
        #perform PCA transformation
        frames = ReadXYZ((filename + 'trajectory.xyz'), N=N, N_types=N_types, shuffle_data=shuffle_data)
        features = TrajectoryToFeatures(frames, N_nn=N_nn, method='distance', step=step)
        transformed_features = incpca.transform(features)
        op = mean(transformed_features, axis=0)
        OPs.append(op)
        OPs_file_stream.write('{},{},{},{},{},{},{}\n'.format(eta,op[0],op[1],op[2],op[3],op[4],op[5]))
        
        print 'Wrote PCA analysis for dataset {}\n'.format(i)

#close the final filestream
OPs_file_stream.close()

#save the OP data
print 'Saving the OP data...\n'
with open('{}processed_data/OPs.pkl'.format(base), 'wb') as file:
    pickle.dump(OPs, file)
        
print 'Finished!!!'
        