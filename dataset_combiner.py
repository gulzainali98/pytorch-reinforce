import h5py
import glob
import numpy as np


def swapPositions(list, pos1, pos2):

    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list

arr=["video_14","video_16","video_21","video_5","video_8"]
check=False

files= glob.glob("datasets/*.h5")
files= swapPositions(files,0,1)
print(files)
hj()
count=0
combined=h5py.File("combined.h5", "r")
print(len(combined.keys()))
print(np.array(combined['video_1']['features'][...]).shape)
hj()
for path in files:
    f= h5py.File(path, "r")
    print(path)
    if "summe" in path:
        check=True
    else:
        check=False
    for vid in f.keys():

        if check:
            # make sure video names in arr are not included
            if vid in arr:
                print(vid)
                continue
            combined.create_dataset('video_'+str(count+1)+"/features",data= np.array(f[vid]['features'][...]))
            count= count+1
        else:
            combined.create_dataset('video_'+str(count+1)+"/features",data= np.array(f[vid]['features'][...]))
            count= count+1
    # print(f.keys())

    # print(f['video_1'].keys())
print(combined.keys())
