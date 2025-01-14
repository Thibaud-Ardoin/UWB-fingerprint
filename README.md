# Exploring UWB Devices Fingerprinting with Deep Learning-Based Authentication
This repository provides a PyTorch implementation and pretrained models for Radio frequency Fingerprinting (RFF) of Ultra-wideband (UWB) devices as described in the paper <a href="https://arxiv.org/abs/2501.04401">Tracking UWB Devices Through Radio Frequency Fingerprinting Is Possible</a>.

The dataset **RUFF** (Rotating UWB For Fingerprint) attached to this work can be downloaded [here](https://zenodo.org/records/11083153).

---

<p align="center">
  <img src="data/images/Pipeline_rff.jpg?raw=true" width="90%">
</p>


## Installation
``conda install --yes --file requirements.txt``

## Training
To train a model on a desired configuration. For exemple a light weighted Vision Transformer model: 

``python src/main.py --config=data/pre-trained-models/ViT_scenario2/config.yaml``


## Evaluation

For evaluation, use this [Notebook](src/model_explorer.ipynb) to run tests of models in different scenarios.


### Models

 <table>
  <tr>
    <th>Model</th>
    <th>link</th>
    <th>config file</th>
    <th>Scenario</th>
    <th> F1 score </th>
    <th> CMC </th>
    <th> AUROC </th>
  </tr>
  <tr>
    <td> ViT </td>
    <td><a href="data/pre-trained-models/ViT_scenario1/model.pth">weights</a></td>
    <td><a href="data/pre-trained-models/ViT_scenario1/config.yaml">config file</a></td>
    <td align="center"> 1 </td>
    <td> 99.9% </td>
    <td> 99.7% </td>
    <td> 0.99 </td>
</tr>
  <tr>
    <td> ViT </td>
    <td><a href="data/pre-trained-models/ViT_scenario2/model.pth">weights</a></td>
    <td><a href="data/pre-trained-models/ViT_scenario2/config.yaml">config file</a></td>
    <td align="center"> 2 </td>
    <td> 64.6% </td>
    <td> 53.4% </td>
    <td> 0.92 </td>
</tr>
  <tr>
    <td> ViT </td>
    <td><a href="data/pre-trained-models/ViT_scenario3/model.pth">weights</a></td>
    <td><a href="data/pre-trained-models/ViT_scenario3/config.yaml">config file</a></td>
    <td align="center"> 3 </td>
    <td> 34.9% </td>
    <td> 18.9% </td>
    <td> 0.76 </td>
</tr>
<tr>
    <td> </td>
    <td>  </td>
    <td>  </td>
    <td>  </td>
    <td>  </td>
    <td>  </td>
    <td> </td>
</tr>
<tr>
    <td> CNN </td>
    <td><a href="data/pre-trained-models/CNN_scenario1/model.pth">weights</a></td>
    <td><a href="data/pre-trained-models/CNN_scenario1/config.yaml">config file</a></td>
    <td align="center"> 1 </td>
    <td> 96.4% </td>
    <td> 86.1% </td>
    <td> 0.93 </td>
  </tr>
<tr>
    <td> CNN </td>
    <td><a href="data/pre-trained-models/CNN_scenario2/model.pth">weights</a></td>
    <td><a href="data/pre-trained-models/CNN_scenario2/config.yaml">config file</a></td>
    <td align="center"> 2 </td>
    <td> 61.5% </td>
    <td> 52.8% </td>
    <td> 0.89 </td>
  </tr>
<tr>
    <td> CNN </td>
    <td><a href="data/pre-trained-models/CNN_scenario3/model.pth">weights</a></td>
    <td><a href="data/pre-trained-models/CNN_scenario3/config.yaml">config file</a></td>
    <td align="center"> 3 </td>
    <td> 14.0% </td>
    <td> 16.9% </td>
    <td> 0.63 </td>
  </tr>
</table> 

### Data splitting Scenarios

<p align="center">
  <img src="data/images/Scenarios3.jpg?raw=true" width="90%">
</p>


## Code Organisation

<p align="center">
  <img src="data/images/map.jpg?raw=true" width="90%">
</p>
