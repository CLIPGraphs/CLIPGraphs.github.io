import torch
import argparse
import numpy as np
from utils import get_room_names, get_mAP, calculate_statistics, get_object_names
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(lang_model):
    obj_names = get_object_names()
    lang_model_parse = lang_model.replace('/', '_')
    relationships = np.load('all_obj_rel.npy', allow_pickle=True).item()
    room_names = get_room_names()
    room_embs = torch.load(f'input_embeddings/room_{lang_model_parse}.pt')

    obj_embs = torch.load(f'input_embeddings/all_objs_{lang_model_parse}.pt')
    print("Language Model: ", lang_model)
    mAP = get_mAP(obj_embs, room_embs, obj_names, room_names, relationships)
    # print(mAP)

    hit, top_3_hit, top_5_hit = calculate_statistics(
        obj_embs, room_embs, obj_names, room_names, relationships, filename=f'{lang_model_parse}_output.txt')

    with open(f'{lang_model_parse}_mAP.txt', 'w') as f:
        f.write(str(mAP))
        f.write('\n')
        f.write(str(hit))
        f.write('\n')
        f.write(str(top_3_hit))
        f.write('\n')
        f.write(str(top_5_hit))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang_model', type=str, default='glove')
    args = parser.parse_args()
    lang_model = args.lang_model

    main(lang_model)
