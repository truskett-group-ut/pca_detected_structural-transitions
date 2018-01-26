from numpy import arange, array

#down sample frames roughly to desired amount
def DownSampleFrames(frames, max_frames):
    num_frames = len(frames)
    if max_frames < num_frames:
        #even sampling
        step = int(float(num_frames)/float(max_frames))
        sampled = []
        for i in arange(0, num_frames, step):
            sampled.append(frames[i])

        #check the sampling
        num_sampled = len(sampled)
        if num_sampled > max_frames:
            sampled = sampled[:max_frames]
        elif num_sampled < max_frames:
            raise Exception('Not enough frames to work with!!!')
        return sampled
    else:
        return frames
       