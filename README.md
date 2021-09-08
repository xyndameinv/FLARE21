# FLARE21

# Experimental environment：
Cuda 10.1 cudnn 7.6.0 python 3.7 tensorflow-gpu 2.2.0 keras 2.3.1
# How to use code：
1.Data process for training：  
（1）Copy origin images and masks to ’ ./data/Raw’  
（2）Run process0.py   
（3）Run process1.py  
（4）Run process2.py  

2.Train shape constrained network（SRM）  
  Prepare data：run process3.py  
  train：run .\SRM\train.py  
3.train binary segmentation network for crop（Crop_Net）   
  Prepare data：run process4.py  
  train: run .\Crop_Net\train.py  
4.Train final multi_organ_network（ISENet）  
（1）Replace the path r'.\srm_model.hdf5' in .\Inception_att_resnet\metrics.py to your path of trained SRM model  
（2）run .\Inception_att_resnet\train.py  
5. Test  
（1）Replace the path r'..\data\Raw\ValidationImg' in .\Inception_att_resnet\predict.py(230 line) to your path of test data, and replace the path r'.\model.20-0.02.hdf5' (231 line) to your path of trained ISENet model, replace the path r'.\crop_model.hdf5' (232 line) to your path of trained Crop_Net model, replace the path r'H:\mutil_CT_seg\result_x1\incep_att_resnet13\ValPrediction_ori2'(300, 301, 302 lines) to your path for saving test result  
（2）run .\Inception_att_resnet\predict.py  
