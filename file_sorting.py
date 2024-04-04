import pdb
import os
import netCDF4 as nc
import numpy as np
test_dir = '/mnt/Data/ai4arctic/test/'
train_dir = '/mnt/Data/ai4arctic/train/'
test_files = os.listdir(test_dir)
train_files = os.listdir(train_dir)
distance_dict = []
for fname in test_files:    
    print('testfile:',fname)    
    test_data = nc.Dataset(test_dir+fname)    
    long_list = test_data['sar_grid_longitude'][:]    
    lat_list = test_data['sar_grid_latitude'][:]    
    center_long = (max(long_list)+min(long_list))/2    
    center_lat = (max(lat_list)+min(lat_list))/2    
    center_test = np.array((center_long, center_lat))    
    #print('test center:',center_long,',',center_lat)    
    distance_list = []    
    for fname_train in train_files:        
        train_data = nc.Dataset(train_dir + fname_train)        
        long_list_train = train_data['sar_grid_longitude'][:]        
        lat_list_train = train_data['sar_grid_latitude'][:]        
        center_long_train = (max(long_list_train) + min(long_list_train)) / 2        
        center_lat_train = (max(lat_list_train) + min(lat_list_train)) / 2        
        center_train = np.array((center_long_train, center_lat_train))        
        #print('train center:', center_long_train, ',', center_long_train)        
        distance = np.linalg.norm(center_test - center_train)        
        #pdb.set_trace()        
        distance_list.append((distance,fname_train))    
    #pdb.set_trace()    
    dis_array = np.array(distance_list, dtype=[('x', float), ('y', 'S200')])    
    #pdb.set_trace()    
    sorted_dis = np.sort(dis_array,order='x')    
    sorted_dis = np.sort(dis_array,0)    
    distance_dict.append(sorted_dis)    
    #dict = {'distance': sorted_dis[:,0], 'test_file': sorted_dis[:,1]}    
    # #df = pd.DataFrame(dict)    
    np.savetxt(fname.split('.')[0] + '_ori.csv', dis_array, fmt='%s')    
    np.savetxt(fname.split('.')[0]+'.csv', sorted_dis, fmt='%s')