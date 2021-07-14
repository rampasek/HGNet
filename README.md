# Hierarchical Graph Net

![HGNet-viz](./HGNet-edgepool.png)

Graph neural networks (GNNs) based on message passing between neighboring nodes are known to be insufficient for capturing long-range interactions in graphs.
In this project we study hierarchical message passing models that leverage a multi-resolution representation of a given graph. This facilitates learning of features that span large receptive fields without loss of local information, an aspect not studied in preceding work on hierarchical GNNs. 
We introduce Hierarchical Graph Net (HGNet), which for any two connected nodes guarantees existence of message-passing paths of at most logarithmic length w.r.t. the input graph size. Yet, under mild assumptions, its internal hierarchy maintains asymptotic size equivalent to that of the input graph. We observe that our HGNet outperforms conventional stacking of GCN layers particularly in molecular property prediction benchmarks. Finally, we propose two benchmarking tasks designed to elucidate capability of GNNs to leverage long-range interactions in graphs.

## Dependencies

Create a Python3 venv with:
* python>=3.7
* [PyTorch](http://pytorch.org/)>=1.7
* [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)>=1.6.3
* numpy
* sklearn
* tqdm
* [TensorBoard](https://www.tensorflow.org/tensorboard)>=2.3


## Training

The datasets will be automatically downloaded via PyTorch Geometric (or generated) to `./data/`.

For datasets that do not provide standardized train/validation/test splits, data split from `./splits/` will be used for cross-validation.

Model hyperparameters for each dataset are loaded from config files in `./configs/`, but can be over-ridden by command-line arguments.

To train default GNN architecture on default dataset (without Tensorboard logging) simply run:
```bash
python main.py --no_tensorboard
```

To specify the GNN model and dataset (with Tensorboard logging) use cmd-line args, e.g.:
```bash
python main.py --dataset=Cora --model=HGNet --depth=1
```

Progression of training & validation statistics can be viewed in TensorBoard, e.g.:
```bash
tensorboard --logdir logs/run_20210608T052342
```