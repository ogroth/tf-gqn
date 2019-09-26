# tf-gqn
![Flying through a scene based on GQN view interpolation](gqn_demo.gif)

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
The major software requirements can be installed on an Ubuntu machine via:

```bash
$ sudo apt-get install python3-pip python3-dev virtualenv
```

The code requires at least **Tensorflow 1.12.0**.
Also, in order to run the models efficiently on GPU, the latest NVIDIA drivers, CUDA and cuDNN frameworks which are compatible with Tensorflow should be installed  ([see version list](https://www.tensorflow.org/install/install_sources#tested_source_configurations)).


## Installation
All Python dependencies should live in their own virtual environment. All runtime requirements can be easily installed via the following commands:

```bash
$ virtualenv -p python3 venv
$ source venv/bin/activate
(venv) $ pip3 install -r requirements.txt
```

## Training
### Training Data

The [data provider](data_provider/gqn_provider.py) implementation is adapted from [https://github.com/deepmind/gqn-datasets](https://github.com/deepmind/gqn-datasets) and uses the more up-to-date `tf.data.dataset` input pipeline approach.

The training datasets can be downloaded from: [https://console.cloud.google.com/storage/gqn-dataset](https://console.cloud.google.com/storage/gqn-dataset)

To download the datasets you can use the [`gsutil cp`](https://cloud.google.com/storage/docs/gsutil/commands/cp) command; see also the `gsutil` [installation instructions](https://cloud.google.com/storage/docs/gsutil_install).

### Training Script
The [training script](train_gqn.py) can be started with the following command, assuming the GQN datasets are located in `data/gqn-dataset`:

```bash
(venv) $ python3 train_gqn.py \
  --data_dir data/gqn-dataset \
  --dataset rooms_ring_camera \
  --model_dir models/rooms_ring_camera/gqn
```

For more verbose information (and tensorboard summaries), you can pass the  `--debug` option to the script as well.

### Tensorboard Summaries
When the `--debug` flag is passed to the training script, image summaries will be written to the tensorboard records.
During the training phase of the network, results from the inference network will be shown (`target_inference`). These images will resemble the target images relatively quickly but are _not_ indicative of model performance because they are computed with the posterior from the target images which are only available during training phase.
During the evaluation phase of the network, results from the generator network will be shown (`target_generation`). These visual results indicate how well the GQN performs when deployed in prediction mode.

## Deployment

### tf.Estimator
Using the tf.estimator API is the most basic form of using the GQN model.
An estimator can be set up by passing the `gqn_draw_model_fn`, the model parameters and the path to the model directory with a corresponding snapshot. An [example](train_gqn.py#L170) can be found in the training script.
Once the estimator is instantiated, it can be trained further (`model.train()`) or used for evaluation or prediction purposes (`model.eval()` or `model.predict()`).
In evaluation and prediction mode, the generator is used.

### GqnViewPredictor
We provide a convenience wrapper around the `tf.estimator` with the [GqnViewPredictor class](gqn/gqn_predictor.py#L33).
The view predictor can be set up by pointing to a model directory containing a model config (`gqn_config.json`) and a corresponding snapshot.
The predictor features APIs to [add new context frames](gqn/gqn_predictor#L102) and [render a query view](gqn/gqn_predictor#L128) based on the currently loaded context.
An example application of the view predictor class can be found in the [view interpolation notebook](notebooks/view_interpolation.ipynb).

### Model Snapshots
Model snapshots for the following GQN datasets are available:
- [`shepard_metzler_5_parts.tar.gz`](https://shapestacks.robots.ox.ac.uk/static/download/tf-gqn/models/shepard_metzler_5_parts.tar.gz)
- [`shepard_metzler_7_parts.tar.gz`](https://shapestacks.robots.ox.ac.uk/static/download/tf-gqn/models/shepard_metzler_7_parts.tar.gz)

In order to use a snapshot, just download the archive and unpack it into the 'models' sub-directory of this repository (which is the default path for all scripts and notebooks to use them).

Each snapshot directory also contains `*-runcmd.json` and `gqn_config.json` files detailing all training settings and model hyper-parameters. You can also run `tensorboard` on the `models` directory to display all summaries which have been tracked during the model training runs.

## Jupyter Notebooks
Jupyter notebooks for running examples of the data loader and view predictor can be found under [notebooks/](notebooks) and a jupyter server can be started with:
```bash
(venv) $ cd notebooks/
(venv) $ jupyter notebook
```

### Dataset Viewer
The [dataset viewer notebook](notebooks/gqn_dataset.ipynb) illustrates the use of the [gqn_input_fn](data_provider/gqn_provider.py#L189) and can be used to browse through the different GQN datasets.

### View Interpolation Experiment
The [view interpolation notebook](notebooks/view_interpolation.ipynb) illustrates the use of a [GqnViewPredictor](gqn/gqn_predictor.py#L33) and can be used to render an imagined flight through a scene as shown in [DeepMind's blog post](https://deepmind.com/blog/neural-scene-representation-and-rendering/).

## Notes
A few random notes about this implementation:

- We were not able to train the model with the learning rate scheme reported in the original paper (from 5\*10e-4 to 5\*10e-5 over 200K steps). This always resulted in a local minimum only generating light blue sky and a grey blob of background. We achieved good results by lowering all learning rates by one order of magnitude.
- Currently, our implementation does not share the convolutional cores between the inference and generation LSTMs. With shared cores we observed the KL divergence between posterior and prior collapsing to zero frequently and obtained generally inferior results (which is in line with the results reported in the paper).
- In our tests, we found eight generation steps to be a good trade-off between training stability, training speed and visual quality.

## Authors

- Oliver Groth [[github](https://github.com/ogroth)]
- Ștefan Săftescu [[github](https://github.com/SliMM)]

Done during our PhD research at the [Oxford Robotics Institute](http://ori.ox.ac.uk/), and the [Visual Geometry Group](http://www.robots.ox.ac.uk/~vgg/).