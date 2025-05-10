# CS4782 (Deep Learning) Final Project

Re-implementation of the paper: [A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730)

## Introduction

## Chosen Result

## GitHub Contents

## Re-implementation Details

### Self-supervised Experiment with Masking Ratio
We implemented an extension to the [PatchTST self-supervised model](https://github.com/yuqinie98/PatchTST/tree/204c21efe0b39603ad6e2ca640ef5896646ab1a9) which uses a randomized mask ratio between 0.2 and 0.6 during each forward pass. For every batch, a new mask ratio was uniformly sampled from this range and used to randomly select a proportion of patch embeddings to be masked. These masked patches were then replaced with learnable mask tokens, and the model was trained to reconstruct the original values. Aside from this randomized masking strategy, all other aspects of the pretraining and evaluation process—such as loss functions, optimizer settings, and downstream fine-tuning—remained consistent with the original PatchTST setup. Since the task was to reconstruct the missing patches, no attention rescaling or architectural modifications were required. We evaluated the MSE, MAE, and training and validation losses using the ETTH1 and ETTH2 datasets. 


## Reproduction Steps

### Self-Supervised Experiment with Masking Ratio
First, install the requirements. 

```bash
% cd code/PatchTST_self_supervised
% pip install -r requirements.txt
```

The datasets ``etth1`` and ``etth2`` are already downloaded in ``code/PatchTST_self_supervised/saved_models/`` from  [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). Run the following script, which performs pretraining and finetuning with both fixed mask ratio and random mask ratio on the ``etth1`` and ``etth2`` datasets:

```bash
% ./script.sh
```

The script output will be dumped in ``code/PatchTST_self_supervised/run.logs``. The performance of the model will be in ``code/PatchTST_self_supervised/saved_models/etth2/masked_patchtst/based_model`` and ``code/PatchTST_self_supervised/saved_models/etth1/masked_patchtst/based_model``. The data for fixed mask ratio experiments is dumped in the csv files labeled with ``model1`` and the data for random mask ratio experiments is dumped in files labeled ``model2``.

## Results/Insights

## Conclusion

## References

## Acknowledgements