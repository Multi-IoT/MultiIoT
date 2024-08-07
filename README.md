# MultiIoT: Benchmarking Machine Learning for the Internet of Things

This repo contains the data and code for MultiIoT.


## Contributors

Correspondence to: 
  - [Shentong Mo](https://scholar.google.com/citations?user=6aYncPAAAAAJ&hl=en) (shentongmo@gmail.com)
  - [Louis-Philippe Morency](https://www.cs.cmu.edu/~morency/) (morency@cs.cmu.edu)
  - [Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/) (rsalakhu@cs.cmu.edu)
  - [Paul Pu Liang](http://www.cs.cmu.edu/~pliang/) (pliang@cs.cmu.edu)

## Paper

[**IoTLM: Large Multisensory Language Models for the Internet of Things**]()<br>
Shentong Mo*, Ruslan Salakhutdinov, Louis-Philippe Morency, Paul Pu Liang*<br>
arXiv 2024.

[**MultiIoT: Benchmarking Machine Learning for the Internet of Things**](https://arxiv.org/abs/2311.06217)<br>
Shentong Mo*, Louis-Philippe Morency, Ruslan Salakhutdinov, Paul Pu Liang*<br>
arXiv 2023.


If you find this repository useful, please cite our paper:
```
@article{mo2024IoTLM,
  title={IOT-LM: Large Multisensory Language Models for the Internet of Things},
  author={Mo, Shentong and Morency, Louis-Philippe and Salakhutdinov, Ruslan and Liang, Paul Pu},
  journal={arXiv preprint arXiv},
  year={2024}
}

@article{mo2023MultiIoT,
  title={MultiIoT: Benchmarking Machine Learning for the Internet of Things},
  author={Mo, Shentong and Morency, Louis-Philippe and Salakhutdinov, Ruslan and Liang, Paul Pu},
  journal={arXiv preprint arXiv:2312.01017},
  year={2023}
}
```

## Overview

![](/images/overview.png)

The Internet of Things (IoT), the network integrating billions of smart physical devices embedded with sensors, software, and communication technologies for the purpose of connecting and exchanging data with other devices and systems, is a critical and rapidly expanding component of our modern world. The IoT ecosystem provides a rich source of real-world modalities such as motion, thermal, geolocation, imaging, depth, sensors, video, and audio for prediction tasks involving the pose, gaze, activities, and gestures of humans as well as the touch, contact, pose, 3D of physical objects. Machine learning presents a rich opportunity to automatically process IoT data at scale, enabling efficient inference for impact in understanding human wellbeing, controlling physical devices, and interconnecting smart cities. 

To develop machine learning technologies for IoT, this paper proposes MultiIoT, the most expansive IoT benchmark to date, encompassing over 1.15 million samples from 12 modalities and 8 tasks. MultiIoT introduces unique challenges involving (1) learning from many sensory modalities, (2) fine-grained interactions across long temporal ranges, and (3) extreme heterogeneity due to unique structure and noise topologies in real-world sensors. We also release a set of strong modeling baselines, spanning modality and task-specific methods to multisensory and multitask models to encourage future research in multisensory representation learning for IoT.


## Environment Setup


```
conda create -n multiiot python=3.8
conda activate multiiot
pip install requirements.txt
```

## Data Download

1. [TouchPose](https://github.com/eth-siplab/TouchPose) (image, capacitance, depth, pose)
2. [RGBGaze](https://github.com/svip-lab/RGBD-Gaze) (image, Gaze, depth, IMU)
3. [SAMoSA](https://github.com/cmusmashlab/SAMoSA) (audio, IMU)
4. [EyeMU](https://github.com/FIGLAB/EyeMU) (Gaze, IMU)
5. [DIP-IMU](https://github.com/eth-ait/dip18) (image,pose, IMU)
6. [LLVIP](https://github.com/bupt-ai-cz/LLVIP) (image, thermal)
7. [KITTI](https://www.cvlibs.net/datasets/kitti/index.php) (image,camera, GPS, IMU)
8. [Ego4D](https://ego4d-data.org/docs/data/imu/) (video, IMU)

For the easy access, we provide the processed data in the following link:
[MultiIoT Data](https://drive.google.com/drive/folders/1UuWeEYfl_wt2_T36MuP3pjsYzQW6xLEK?usp=sharing)

## Experiments Usage

For the task on activity recognition, we provide a simple example to run the experiments on the unimodal setting. The code is located in the `examples` directory. To run the example, execute the following command:

```
python examples/iot/unimodal_0.py
python examples/iot/unimodal_1.py
```

For running our multisensory model, execute the following command:

```
python examples/iot/multisensory_fusion_0.py
python examples/iot/multisensory_fusion_1.py
```

For running our multisensory multitask model, execute the following command:

```
python examples/iot/multisensory_multitask.py
```

## IoT-LM

The Internet of Things (IoT) network integrating billions of smart physical devices embedded with sensors, software, and communication technologies is a critical and rapidly expanding component of our modern world. The IoT ecosystem provides a rich source of real-world modalities such as motion, thermal, geolocation, imaging, depth, sensors, and audio to recognize the states of humans and physical objects. Machine learning presents a rich opportunity to automatically process IoT data at scale, enabling efficient inference for understanding human wellbeing, controlling physical devices, and interconnecting smart cities. To realize this potential, we introduce IoT-LM, an open-source large multisensory language model tailored for the IoT ecosystem. 


For running our IoT language model, execute the following command:

```
python examples/iot/iotlm.py --batch_size 8 \
    --epochs 100 \
    --split_epoch 50 \
    --warmup_epochs 5 \
    --blr 1.0e-4 \
    --weight_decay 0.05 \
    --llama_path /path/to/llama_model_weights

```


## IoT-LM Demo

[Coming soon!]


## Open for New Datasets and Algorithms


To add a new dataset:

1. Go to datasets/
2. Add a new folder if appropriate
3. Write a python file with a get_dataloader function that returns a tuple of 3 dataloaders (for train, valid, test data respectively) containing preprocessed data. Please following the existing examples (such as samosa: datasets/samosa/get_data.py)
4. Go to examples/ and write an example training python file following the existing examples
5. Check that calling the dataloader and running a simple training script works


To add a new algorithm:

1. Figure out which subfolder to add it into:
- unimodals/ : unimodal architectures
- fusions/ : multimodal fusion architectures
2. see examples/iot/ and write an example training python file following the existing examples
3. check that calling the added functions and running a simple training script works
4. Make sure your new modules are well documented by comments in its input and output format and shapes


We welcome new contributions to MultiIoT through new datasets and algorithms. Please refer to the sections above for instructions on adding new datasets and algorithms, and open a pull request if you would like to see a specific dataset or algorithm added. 


