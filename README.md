# Physics-based 3D Avatar Full-Body Motion Control from Sparse Data and Direction Control using Deep Reinforcement Learning

## Contributions
The contributions of this work are as follow: 
1. Enables a VR user with limited mobility to control a 3D avatar and explore spacious virtual environment
2. Enables object interaction that satisfies physical laws with a key feature of sparse sensor reward

<br/>

## Results and Applications
### Visualization result
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
<br/>

### Application
<p align="center">
  <img src="https://github.com/haaappytoast/22_MS1_Master_Thesis/assets/45995611/72c09f6a-9038-4363-a736-0507f6851825.gif" width="600" height="360">
  <br>
  <em>Blocking a projectile</em>
</p>

<p align="center">
  <img src="https://github.com/haaappytoast/23_MS2_Master_Thesis/assets/45995611/d5a853d2-4539-499b-bf9d-60e8eef6990b" width="600" height="360">
  <br>
  <em>Punching an object</em>
</p>


<br/><br/><br/>

## 1. Dependencies
* IssacGym Pr4<br/>
  * Link: [Webpage](https://developer.nvidia.com/isaac-gym) 
* Composite Motion Learning with Task Control [Xu et al. SIGGRAPH 2023]
  * Link: [Github](https://github.com/xupei0610/CompositeMotion)
  

## 2. Motion Data 
### Motion captured data
* We provide our motion data in ```assets/motions``` and ```assets/retargeted```.
  * Motions located in ```assets/motions``` are provided by Composite Motion Learning with Task Control.
  * The blocking/pickup/punch/throw motions located in ```assets/retargeted``` are extracted from the demo provided by Mixamo.
  * Tennis motions shown in the project is not provided due to the commercial license.
  * Motions extracted from Mixamo are retargeted to our skeleton model with the code provided by [ASE](https://github.com/nv-tlabs/ASE). Please refer to an example retargeting script in ```ase/poselib/retarget_motion.py```.

<p align="center">
  <img src="https://github.com/haaappytoast/22_MS1_Master_Thesis/assets/45995611/e813d22d-3cce-4f6e-a657-f1d9cbd7965c" width="360" height="480">
  <br>
  <em> Motion sequence demo provided by Mixamo. <br/> Block, pick fruits, punch motions are illustrated in the rows from top to bottom.</em>
</p>
<br/> 

<p align="center">
  <img src="https://github.com/haaappytoast/22_MS1_Master_Thesis/assets/45995611/3029a592-7761-41b6-b4c0-f25ac4d42abb" width="360" height="190">
  <br>
  <em> Example demo of tennis motion retargeted to our model</em>
</p>

### Real VR user motion dataset
<p align="center">
  <img src="https://github.com/haaappytoast/23_MS2_Master_Thesis/assets/45995611/1d92fc1f-fc76-426b-97a8-bb601b0b5f74" width="480" height="160">
  <br>
  <em> Overview of collecting VR user motion data</em>
</p>

  **Data Acquisition from Movement SDK**
  * Real sensor data of VR headsets and motions of A VR user are extracted using [MovementSDK](https://developer.oculus.com/documentation/unity/move-overview/) and Unity3D.

  **Retargeting process**
  
* Same retargeting method used in motion captured data is utilized to transfer the motion of real VR user's motion onto our skeleton model of IsaacGym.

<br/> 




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


