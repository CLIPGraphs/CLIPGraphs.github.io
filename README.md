# CLIPGraphs

This repository contains the code for obtaining the results for [CLIPGraphs: Multimodal graph networks to infer object-room affinities for scene rearrangement](https://clipgraphs.github.io)

To reproduce the results corresponding to CLIP ViT-H/14 model:
```
python get_results.py --split test
python get_results.py --split val
```

This will print the statistics for the model with different metrics and generate a new file called `obj_room.txt` containing the predicted object-room mappings.