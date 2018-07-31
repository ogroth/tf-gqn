# tf-gqn
![Flying through a scene generated from a single context image](gqn_demo.gif)

This repository contains a Tensorflow implementation of the *Generative Query Network (GQN)* described in 'Neural Scene Representation and Rendering' by Eslami et al. (2018).

## Original paper

**Neural Scene Representation and Rendering** [[PDF][]] [[blog][]]

S. M. Ali Eslami, Danilo J. Rezende, Frederic Besse, Fabio Viola, Ari S. Morcos,
Marta Garnelo, Avraham Ruderman, Andrei A. Rusu, Ivo Danihelka, Karol Gregor,
David P. Reichert, Lars Buesing, Theophane Weber, Oriol Vinyals, Dan Rosenbaum,
Neil Rabinowitz, Helen King, Chloe Hillier, Matt Botvinick, Daan Wierstra,
Koray Kavukcuoglu and Demis Hassabis

[pdf]: https://deepmind.com/documents/211/Neural_Scene_Representation_and_Rendering_preprint.pdf
[blog]: https://deepmind.com/blog/neural-scene-representation-and-rendering/

If you use this repository, please cite the original publication:

```
@article{eslami2018neural,
  title={Neural scene representation and rendering},
  author={Eslami, SM Ali and Rezende, Danilo Jimenez and Besse, Frederic and Viola, Fabio and Morcos, Ari S and Garnelo, Marta and Ruderman, Avraham and Rusu, Andrei A and Danihelka, Ivo and Gregor, Karol and others},
  journal={Science},
  volume={360},
  number={6394},
  pages={1204--1210},
  year={2018},
  publisher={American Association for the Advancement of Science}
}
```

## Software Requirements
The code requires at least **Tensorflow 1.8.0**. It has been tested on the following platforms:

- Ubuntu 16.04 kernel 4.15.0-24-generic with Python 3.5.2;
- macOS Sierra 10.12.5 with Python 3.6.5
- macOS High Sierra 10.13.6 with Python 3.5.2.

	
The major software requirements can be installed on an Ubuntu machine via:

```bash
$ sudo apt-get install python3-pip python3-dev virtualenv
```

Also, in order to run the models efficiently on GPU, the latest NVIDIA drivers, CUDA and cuDNN frameworks which are compatible with Tensorflow should be installed  ([see version list](https://www.tensorflow.org/install/install_sources#tested_source_configurations)).


## Installation
All Python dependencies should live in their own virtual environment. All runtime requirements can be easily installed via the following commands:

```bash
$ virtualenv -p python3 venv
$ source venv/bin/activate
(venv) $ pip3 install -r requirements.txt
```

Additional requirements for development purposes can be found in ```dev_requirements.txt``` and can be added on demand.

```bash
(venv) $ pip3 install -r dev_requirements.txt
```

## Training
### Training Data

The data provider implementation is adapted from: [https://github.com/deepmind/gqn-datasets]()

The training datasets can be downloaded from: [https://console.cloud.google.com/storage/gqn-dataset]()

To download the datasets you can use the [`gsutil cp`](https://cloud.google.com/storage/docs/gsutil/commands/cp) command; see also the `gsutil` [installation instructions](https://cloud.google.com/storage/docs/gsutil_install).

### Training Script
The training script can be started with the following command, assuming the GQN datasets have been downloaded in `/tmp/data/gqn-dataset`:

```bash
python3 train_gqn_draw.py \
  --data_dir /tmp/data/gqn-dataset \
  --dataset rooms_ring_camera \
  --model_dir /tmp/models/gqn
```

For more verbose information (and summaries), you can pass the  `--debug` option to the script as well.

## Notes
A few random notes about this implementation:

- The model has so far only been trained successfully on the ```rooms_ring_camera``` dataset of the GQN data repository.
- We were not able to train the model with the learning rate scheme reported in the original paper (from 5\*10e-4 to 5\*10e-5 over 200K steps). This always resulted in a local minimum only generating light blue sky and a grey blob of background. We achieved good results by lowering all learning rates by one order of magnitude.
- Currently, our implementation does not share the convolutional cores between the inference and generation LSTMs. With shared cores we observed the KL divergence between posterior and prior collapsing to zero frequently and obtained generally inferior results (which is in line with the results reported in the paper).
- In our tests, we found eight generation steps to be a good trade-off between training stability, training speed and visual quality.
- We have trained models on Titan Xp and GTX 1080Ti GPUs usually obtaining visually reasonable results after about one day of training.

## Authors

- Oliver Groth [[github](https://github.com/ogroth)]
- Ștefan Săftescu [[github](https://github.com/SliMM)]

Done during our PhD research at the [Oxford Robotics Institute](http://ori.ox.ac.uk/), and the [Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/).