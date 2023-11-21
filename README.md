# 22_MS1_Master_Thesis


## Contributions
The contributions of this work are as follow: 
1. Enables a VR user with limited lower-body mobility to control full-body movement of a physics-based 3D avatar
2. Introduces sparse sensor input reward to realistically control upper body movements of a 3D avatar
<br/><br/><br/>

<p align="center">
  <img src="https://github.com/haaappytoast/22_MS1_Master_Thesis/assets/45995611/7f4fe98e-7cc2-4608-b7fe-39d40fde4d2c">
  <br>
  <em> (Top) Motion sequences of the VR user performing throwing motion with direction control. (Bottom) Reconstructed full-body motion of the 3D avatar.</em>
</p>

<p align="center">
  <img src="https://github.com/haaappytoast/22_MS1_Master_Thesis/assets/45995611/993a0b91-35ce-4d23-8a53-4cc0e92236c1">
  <br>
  <em>(Top) Motion sequences of the VR user performing punch motion with direction control. (Bottom) Reconstructed full-body motion of the 3D avatar.</em>
</p>


<br/><br/><br/>

## 1. Dependencies
* IssacGym Pr4<br/>
  * Link: [Webpage](https://developer.nvidia.com/isaac-gym) 
* Composite Motion Learning with Task Control [Xu et al. SIGGRAPH 2023]
  * Link: [Github](https://github.com/xupei0610/CompositeMotion)
  

## 2. Motion Data 
* We provide our motion data in ```assets/motions``` and ```assets/retargeted```.
  * Motions located in ```assets/motions``` are provided by Composite Motion Learning with Task Control.
  * The blocking/pickup/punch/throw motions located in ```assets/retargeted``` are extracted from the demo provided by Mixamo.
  * Tennis motions shown in the project is not provided due to the commercial license.
  * Motions extracted from Mixamo are retargeted to our skeleton model with the code provided by [ASE](https://github.com/nv-tlabs/ASE). Please refer to an example retargeting script in ```ase/poselib/retarget_motion.py```.

<br/> 

<p align="center">
  <img src="https://github.com/haaappytoast/22_MS1_Master_Thesis/assets/45995611/e813d22d-3cce-4f6e-a657-f1d9cbd7965c" width="480" height="600">
  <br>
  <em> Motion sequence demo provided by Mixamo. <br/> Block, pick fruits, punch motions are illustrated in the rows from top to bottom.</em>
</p>
<br/> 

<p align="center">
  <img src="https://github.com/haaappytoast/22_MS1_Master_Thesis/assets/45995611/3029a592-7761-41b6-b4c0-f25ac4d42abb" width="600" height="250">
  <br>
  <em> Example demo of tennis motion retargeted to our model</em>
</p>


## 3. Policy Training 
```
python main.py <configure_file> --ckpt <checkpoint_dir>
```
- Example code of training punching motion
  * options
    * ```--headless```: Run headless without creating a viewer window
    * ```--server```: which server you are running the code
    * Others are stated in ```main.py```
```
python main.py config/punch_joystick.py --ckpt 1121_punch_MIX --headless --server local
```

## 4. Policy Evaluation 
```
python main.py <configure_file> --ckpt <checkpoint_dir> --test
```
- Example code of testing punching motion
```
python main.py config/punch_joystick.py --ckpt 1121_punch_MIX/ckpt-20000 --test
```



