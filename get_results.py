import os
import json
import torch
from model import GCN
from utils import *
import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix


def main(image_model_type, lang_model_type, model_name, loss, split):
    rooms = get_room_names()
    objects = get_object_names()
    categories = get_category_names()
    input_dim = 1024
    output_dim = 128
    image_model_type_parse = image_model_type.replace('/', '_')
    lang_model_type_parse = lang_model_type.replace('/', '_')

    relationships = np.load('all_obj_rel.npy', allow_pickle=True).item()
    model = GCN(input_dim, output_dim).to("cuda")
    # model = DataParallel(model)
    model.load_state_dict(torch.load(
        f'output/{model_name}/{loss}/prompt_const_15_k_40_20_k_20_global_best_val_model_{image_model_type_parse}_{lang_model_type_parse}.pth'))

    color_dict = {
        "bathroom": "#6495ED",
        "bedroom": "#8B008B",
        "childs_room": "#FFA07A",
        "closet": "#F08080",
        "corridor": "#FFD700",
        "dining_room": "#ADFF2F",
        "exercise_room": "#00FFFF",
        "garage": "#808080",
        "home_office": "#9370DB",
        "kitchen": "red",
        "living_room": "#FFC0CB",
        "lobby": "#87CEFA",
        "pantry_room": "#4169E1",
        "playroom": "#FF69B4",
        "storage_room": "#228B22",
        "television_room": "#FF4500",
        "utility_room": "green"
    }

    val_graph = nx.Graph()
    room_features = np.load(
        f'input_embeddings/room_{lang_model_type_parse}_{input_dim}.npy', allow_pickle=True).item()
    obj_features1 = np.load(
        f'input_embeddings/train_110_input/train110_{image_model_type_parse}_features.npy', allow_pickle=True).item()
    obj_features2 = np.load(
        f'input_embeddings/val_27_input/val27_{image_model_type_parse}_features.npy', allow_pickle=True).item()
    obj_features3 = np.load(
        f'input_embeddings/test_131_input/test131_{image_model_type_parse}_features.npy', allow_pickle=True).item()
    print("Found Image Embeddings")
    features = {**room_features, **obj_features1,
                **obj_features2, **obj_features3}
    # for room in rooms:
    #     val_graph.add_node(
    #         room,
    #         name=room,
    #         features=np.reshape(features[room].cpu().detach().numpy(), (input_dim,))
    #         )

    with open('keys.json', 'r') as f:
        keys = json.load(f)

    for key in keys:
        category, object_name_parse, split, name = key
        color = next((color_dict[room] for room in relationships.keys(
        ) if object_name_parse in relationships[room]), None)
        val_graph.add_node(name, name=name, features=np.reshape(
            features[category][object_name_parse][split][name].cpu().detach().numpy(), (input_dim,)), color=color)

    import pdb
    pdb.set_trace()
    self_edges = [(node, node) for node in val_graph.nodes()]
    val_graph.add_edges_from(self_edges)

    val_info2 = nx.get_node_attributes(val_graph, "color")
    # Validation Set
    adjacency_matrix = torch.from_numpy(
        nx.to_numpy_array(val_graph)).to("cuda")
    adjacency_matrix[adjacency_matrix != 0] = 1
    adjacency_matrix_binary = adjacency_matrix.int()
    non_zero_entries = torch.nonzero(adjacency_matrix_binary)
    val_edge_index = torch.empty(
        2, non_zero_entries.shape[0], dtype=torch.long).to("cuda")
    val_edge_index[0, :] = non_zero_entries[:, 0]
    val_edge_index[1, :] = non_zero_entries[:, 1]  # (2,num of edges)
    adjacency_matrix = torch.from_numpy(
        nx.to_numpy_array(val_graph)).to("cuda")
    coo = coo_matrix(adjacency_matrix.cpu().numpy())
    val_edge_weight = torch.tensor(coo.data, dtype=torch.float).cuda()

    val_features = [val_graph.nodes[node]['features']
                    for node in val_graph.nodes()]
    val_features = np.reshape(val_features, (len(val_features), input_dim))
    val_features = torch.from_numpy(val_features).cuda().float()
    print(val_features.shape)

    model.eval()
    with torch.no_grad():
        val_emb = model(val_features.float(), val_edge_index, val_edge_weight)

    # visualize(val_emb, color=list(val_info2.values()), filename=f'val_tsne.png')

    # room_embeddings = torch.stack([val_emb[_] for _ in range(len(rooms))])

    room_dict = np.load(
        f'output_room_embeddings/{model_name}/{loss}/prompt_const_15_k_40_20_k_20_local_best_{image_model_type_parse}_{lang_model_type_parse}_room_embedding.npy', allow_pickle=True).item()

    room_embeddings = []
    room_names = []
    for name, em in room_dict.items():
        room_embeddings.append(em)
        room_names.append(name)

    room_embeddings = torch.stack(room_embeddings)
    import pdb
    pdb.set_trace()
    calculate_statistics(val_emb, room_embeddings, list(val_graph.nodes),
                         rooms, relationships)
    print(get_mAP(val_emb, room_embeddings, list(val_graph.nodes),
                  rooms, relationships))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_model', type=str,
                        default='clip_convnext_base')
    parser.add_argument('--lang_model', type=str, default='clip_convnext_base')
    parser.add_argument('--model', type=str, default='GCN')
    parser.add_argument('--loss', type=str, default='margin')
    parser.add_argument('--split', type=str, default='test')

    args = parser.parse_args()
    image_model_type = args.image_model
    lang_model_type = args.lang_model
    model = args.model
    loss = args.loss
    split = args.split

    main(image_model_type, lang_model_type, model, loss, split)
