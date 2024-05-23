# Code of the paper [Physics-informed geometric regularization of heterogeneous reconstructions in cryo-EM](https://openreview.net/pdf?id=41zNERm0J9).

## Data preparation
1. The MD-computed conformation trajectory files for the ADK and Nsp13 datasets are in `./data`.
2. Convert each conformation of the trajectory to its own `.pdb` file.
3. Run [parakeet](https://github.com/rosalindfranklininstitute/parakeet) on the pdb files using the configuration `data/parakeet_config.yaml`.
4. Crop the produced micrographs around the particles, and extract the 3D pose of each particle from the output of Parakeet.
5. Put the resulting tensors in a dict with keys `"imgs"`, `"poses"`, `"conf_id"` and save as `picked_particles_<electron_dose>e.pickle`. The classes `AdkDataset` and `CovDataset` can load this file.

Please contact me (`victor.prins [at] outlook [dot] com`) if you need help with any of these steps. I can also provide the complete preprocessed image datasets (~20GB in total) that were used for the paper.


## Installation
Install the dependencies with `pip install -r requirements-gpu.txt`. The code is tested with Python 3.9. Run `wandb login` on the command line to enable login to your Weights & Biases account.

## Training the network
1. Modify any of the values in `config` in `train.py` as appropriate for your use case.
2. Run `python train.py`.

## Citation
If you find this useful, please cite our paper:
```
@inproceedings{prins2024physics,
  title={Physics-informed geometric regularization of heterogeneous reconstructions in cryo-EM},
  author={Prins, Victor and Diepeveen, Willem and Bekkers, Erik J and {\"O}ktem, Ozan},
  booktitle={ICLR 2024 Workshop on Generative and Experimental Perspectives for Biomolecular Design}
}
```