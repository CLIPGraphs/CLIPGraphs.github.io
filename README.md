# CLIPGraphs

This repository contains the code for obtaining the results for [CLIPGraphs: Multimodal graph networks to infer object-room affinities for scene rearrangement](https://clipgraphs.github.io)

To reproduce the results corresponding to CLIP ViT-H/14 model:
```
python get_results.py --split test
python get_results.py --split val
```

This will print the statistics for the model with different metrics and generate a new file called `GCN_model_output.txt` containing the predicted object-room mappings.


To get the predicted object-room mappings for different language baselines, run the script:
```
python llm_baseline.py --lang_model glove
```
Here, you can replace `glove` with any of the following language models: `roberta`, `glove`, `clip_convnext_base`, `clip_RN50`, or `clip_ViT-H/14`.

This would produce 2 files, 
- `lang_model_mAP.txt` will contain the statistic metrics for the metric, 
- `lang_model_output.txt` will contain the object-room mappings generated by the lang_model.

If you find CLIPGraphs useful for your work please cite:
```
@misc{agrawal2023clipgraphs,
            title={CLIPGraphs: Multimodal Graph Networks to Infer Object-Room Affinities, 
            author={Ayush Agrawal and Raghav Arora and Ahana Datta and Snehasis Banerjee and Brojeshwar Bhowmick and Krishna Murthy Jatavallabhula and Mohan Sridharan and Madhava Krishna}},
            year={2023},
            eprint={2306.01540},
            archivePrefix={arXiv},
            primaryClass={cs.RO}}
```
