# Custom YOLOv5 on RK3588S for OrangePi/OPi 5

This repository provides a straightforward reference for running custom YOLOv5 models on the Neural Processing Unit (NPU) of the OrangePi 5 boards equipped with RK3588 processors.

## Introduction

Rockchip boards offer immense potential for running AI models at speeds comparable to or even faster than Nvidia Jetson boards but at a fraction of the cost and with fewer supply chain issues. While software support from the Radxa team in C++ and Python is still developing and may not yet directly compete with CUDA, the power of open source can accelerate this progress.

In this repository, I compile information from various sources, including several Chinese blogs on CSDN, and provide detailed guides and references. A big thank you to the people who did the trials and published the details online. My aim is to enhance the use of RK3588S and Orangepi for AI applications and contribute to the community's knowledge as the software ecosystem evolves.

Currently This runs demo using the C++ inference code provided by rockchip, I'll try and figure out and add custom model python inferencing with time in this repo.


## Installation & Requirements


1) Installed the ubuntu 22.04 for OPi from [here](https://github.com/Joshua-Riek/ubuntu-rockchip.git). Flashed it using BalenaEtcher ( Should also work for ubuntu 20.04 on OPi)

SBC Board Environment: Ubuntu 22.04 on the Orange pi

Host Machine Environment: Ubuntu 22.04 with asdf installed for different python versions


Chip: RK3588S - Orange Pi 5 (This should also work on RK3588 boards as shown in the reference articles)

## Usage
### A) There are two ways you can setup the host machine for training the model and converting it to rknn.

i) Setup a local virtual environment, but you will have to manage other dependencies
  
  1) Clone the yolov5-6.0 repo from [here](https://github.com/ultralytics/yolov5.git)
  2) Configure a basic python 3.8 virtual environment. I am using [asdf](https://github.com/asdf-vm/asdf.git) for managing different python versions and venv for creating a virtual environment. You can use conda also.

         asdf local python 3.8.10
         python -m venv venv3.8rknn
      
  #### If you are going to run training in the same host machine and have CUDA then run the following else skip this step
   
       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
  Install all yolov5 requirements and the rknn toolkit2 from  [here](https://github.com/rockchip-linux/rknn-toolkit2.git). It is under rknn-toolkit2/packages. You can find toolkit2 wheels for other python versions also
    
        pip install -r requirements.txt
        pip install numpy==1.20.3
        pip install onnx==1.12.0
        pip3 install rknn_toolkit2-1.6.0+81f21f4d-cp38-cp38-linux_x86_64.whl
        
ii) Use Docker to setup the rknn environment so that you do not have to worry about any dependencies. 
  1)(Install docker [here](https://docs.docker.com/engine/install/ubuntu/) and add [user](https://docs.docker.com/engine/install/linux-postinstall/) to run without sudo )
  2) Download this repo and use the dockerfile given.
  
        cd host_machine
        docker build -t rknn_host_docker_1.6 -f Dockerfile_ubuntu_20_04_for_cp38
        docker run -t -i --privileged -v /dev/bus/usb:/dev/bus/usb -v rknn-toolkit2/examples//:/examples -v rknpu2//:/rknpu2 rknn_host_docker_1.6 /bin/bash

  ####  Once its run the first time, you can start and run the docker again if needed using 

        docker start rknn_host_docker_1.6
        docker exec -it rknn_host_docker_1.6 /bin/bash

  3) Clone the yolov5-6.0 repo from [here](https://github.com/ultralytics/yolov5.git)

### B) Test whether the toolkit is running properly by running the yolov5 test script provided by rockchip that runs on yolov5s_relu.onnx

    cd /rknn-toolkit2/onnx/yolov5
    python3 test.py
    #It should end with the following 
    >>......
      ......
      --> Running model
      GraphPreparing : 100%|██████████████████████████████████████████| 153/153 [00:00<00:00, 7477.35it/s]
      SessionPreparing : 100%|█████████████████████████████████████████| 153/153 [00:00<00:00, 499.60it/s]
      done
         class        score      xmin, ymin, xmax, ymax
      --------------------------------------------------
         person       0.884     [ 208,  244,  286,  506]
         person       0.868     [ 478,  236,  559,  528]
         person       0.825     [ 110,  238,  230,  534]
         person       0.339     [  79,  353,  122,  516]
          bus         0.705     [  92,  128,  554,  467]
      Save results to result.jpg!

This is basically running the inference using the rknn model on your host machine, now we will get to  model training and conversion of our custom model to rknn format.
### C) Training and Conversion
  i) First train your model on your custom dataset, taking reference from the [yolov5](https://github.com/ultralytics/yolov5.git) repo.
  ii) Once you have your model best.pt file. we will now convert the model as below:
    a) Modify the models/yolo.py file in the yolov5 directory. Change the forward function in class Detect
    
            # def forward(self, x):
      #     """Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny, nx, 85)`."""
      #     z = []  # inference output
      #     for i in range(self.nl):
      #         x[i] = self.m[i](x[i])  # conv
      #         bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
      #         x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
      #         if not self.training:  # inference
      #             if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
      #                 self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
      #             if isinstance(self, Segment):  # (boxes + masks)
      #                 xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
      #                 xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
      #                 wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
      #                 y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
      #             else:  # Detect (boxes only)
      #                 xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
      #                 xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
      #                 wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
      #                 y = torch.cat((xy, wh, conf), 4)
      #             z.append(y.view(bs, self.na * nx * ny, self.no))
      #     return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
      def forward(self, x):
          z = []  # inference output
          for i in range(self.nl):
              if os.getenv('RKNN_model_hack', '0') != '0':
                  x[i] = torch.sigmoid(self.m[i](x[i]))  # conv
          return x
          
   b) Add the following lines to export.py at the start
   
      import os
      os.environ['RKNN_model_hack'] = 'npu_2'
      
   Replace the line 
   
      shape = tuple((y[0] if isinstance(y, tuple) else y).shape)  # model output shape
   with
   
      shape = tuple(y[0].shape)  # model output shape
      
  c) Now move your best.pt file from the weights folder of your training run to the base directory of yolov5 and run the export command to obtain the corresponding .onnx model with opset 12
  
      ```
        python export.py --weights best.pt --img 640 --batch 1 --include onnx --opset 12
      ```
  d) The model file best.onnx will be in the current folder. Check whether the model is correct in [Netron](https://netron.app/). Click the menu in the upper left corner and then Properties. There should be 3 output layers similar to below
  
  ![image](https://github.com/programmeddeath1/custom_yolov5_npu_rk3588/assets/44861370/1d7264bd-dbcd-47d7-9167-6450e8906ad2)
    
  e) We will now convert the onnx model to rknn.
    Enter /rknpu2/examples/rknn_yolov5_demo/convert_rknn_demo/yolov5  directory and modify onnx2rknn.py. You can open this with the docker option in vscode left menu. Check your rknn_host_docker container and navigate to onnx2rknn.py. 
    Move your converted best.onnx model to this folder
      
  ![image](https://github.com/programmeddeath1/custom_yolov5_npu_rk3588/assets/44861370/09ecafc6-72a1-494e-9a3e-b69398593f36)
      
      -> modify the platform
      -> modify the model path
      -> modify the test image path. (I have taken a sample image from m test dataset)
      -> Modify dataset.txt
       ![image](https://github.com/programmeddeath1/custom_yolov5_npu_rk3588/assets/44861370/5f3fffe2-5172-405d-9613-17b15f6b6b0f)
  f) Run the script to convert onnx to rknn -

          python onnx2rknn.py

  ![image](https://github.com/programmeddeath1/custom_yolov5_npu_rk3588/assets/44861370/9db34e12-f8f2-4915-b25e-fa3751f80fd3)
  
  g) Copy the rknn model to /rknpu2/examples/rknn_yolov5_demo/model/RK3588
  h) Enter /rknpu2/examples/rknn_yolov5_demo_c4/modelthe directory and modify coco_80_labels_list.txt to your custom classes:
  
  ![image](https://github.com/programmeddeath1/custom_yolov5_npu_rk3588/assets/44861370/f3a1e1b4-c56e-4754-971d-b0638cd75a0b)
      
  i) Modify /rknpu2/examples/rknn_yolov5_demo/include/postprocess.h, change the number of categories to the number of custom classes
  
  ![image](https://github.com/programmeddeath1/custom_yolov5_npu_rk3588/assets/44861370/6f3974d4-fd14-4671-a0a8-ec3bd70e5889)
  
  j) Enter the /rknpu2/examples/rknn_yolov5_demo folder on cli and run the build for 3588
  
      chmod +x build-linux_RK3588.sh
      ./build-linux_RK3588.sh
          
  If you get a cmake error about cmake compiler version run the following command and try again
  
      sudo apt-get install gcc-aarch64-linux-gnu
    
  This will create a build/ folder and install/ folder. Now zip this install folder along with your rknn model from /rknpu2/examples/rknn_yolov5_demo/convert_rknn_demo/yolov5/rknn_models folder and your test images. Copy this zip file to your Opi board
#### D) Inference on the OPi
  i) unzip your install/model folder in OPi, and give execute access to install/rknn_yolov5_demo_linux/rknn_yolov5_demo and finally run on your test images.
  
        chmod +x ./rknn_yolov5_demo
        ~/app/rknn-toolkit2-master/board_files/install/rknn_yolov5_demo_Linux$ ./rknn_yolov5_demo ../../yolov5s-640-640_rk3588.rknn ../../two_tetra.jpg
        post process config: box_conf_threshold = 0.25, nms_threshold = 0.45
        Loading mode...
        sdk version: 1.6.0 (9a7b5d24c@2023-12-13T17:31:11) driver version: 0.9.2
        model input num: 1, output num: 3
          index=0, name=images, n_dims=4, dims=[1, 640, 640, 3], n_elems=1228800, size=1228800, w_stride = 640, size_with_stride=1228800, fmt=NHWC, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
          index=0, name=output0, n_dims=4, dims=[1, 36, 80, 80], n_elems=230400, size=230400, w_stride = 0, size_with_stride=307200, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
          index=1, name=332, n_dims=4, dims=[1, 36, 40, 40], n_elems=57600, size=57600, w_stride = 0, size_with_stride=92160, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003917
          index=2, name=334, n_dims=4, dims=[1, 36, 20, 20], n_elems=14400, size=14400, w_stride = 0, size_with_stride=30720, fmt=NCHW, type=INT8, qnt_type=AFFINE, zp=-128, scale=0.003922
        model is NHWC input fmt
        model input height=640, width=640, channel=3
        Read ../../two_tetra.jpg ...
        img width = 1280, img height = 720
        resize image with letterbox
        once run use 25.621000 ms
        loadLabelName ./model/coco_80_labels_list.txt
        tetra_pack @ (576 330 680 570) 0.957017
        tetra_pack @ (396 334 506 574) 0.949204
        save detect result to ./out.jpg
        loop count = 10 , average run  24.682200 ms

  

The average run time per image varies from around 25 to 33ms which gives us about 30-40 FPS  

   
This section will detail the steps required to run custom training and to deploy and test the models on the OrangePi 5 board. Stay tuned for more step-by-step guides and screenshots.

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star if you find it useful! Thanks again!


## References

- [Yolov5 on RK3588 by jcfszxc](https://blog.csdn.net/jcfszxc/article/details/135708013?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-2-135708013-blog-128250096.235%5Ev43%5Epc_blog_bottom_relevance_base2)
- [Custom training YOLOv5 on RK3588](https://blog.csdn.net/m0_62919535/article/details/136788091?spm=1001.2101.3001.6650.6&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-6-136788091-blog-135708013.235%5Ev43%5Epc_blog_bottom_relevance_base6)
- [Running standard YOLOv5 on RK3588 By Qengineering](https://github.com/Qengineering/YoloV5-NPU-Rock-5)
- [YOLOv5 by Ultralytics](https://github.com/ultralytics/yolov5.git)
- [YOLOv5 RK3588 Real-time Camera Inferencing] (https://github.com/Applied-Deep-Learning-Lab/Yolov5_RK3588)
- [rknn toolkit and rknpu](https://github.com/rockchip-linux/rknn-toolkit2.git)

## License

Distributed under the Apache License 2.0. 
