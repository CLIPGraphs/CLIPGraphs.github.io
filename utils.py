import numpy as np
import torch
import re
from sklearn.metrics import precision_score, recall_score, average_precision_score, confusion_matrix, classification_report, precision_recall_curve, top_k_accuracy_score


def get_room_names():
    with open('rooms.txt', 'r') as f:
        data = f.read().splitlines()
    return data


def get_object_names():
    with open('objects.txt', 'r') as f:
        data = f.read().splitlines()
    return data


def get_category_names():
    with open('categories.txt', 'r') as f:
        data = f.read().splitlines()
    return data


def get_mAP(obj_embeddings, room_embeddings, obj_names, room_names, relationships):
    similarity_matrix = np.zeros((len(obj_embeddings), len(room_embeddings)))
    for i, obj_emb in enumerate(obj_embeddings):
        similarity_scores = torch.nn.functional.cosine_similarity(
            obj_emb.unsqueeze(0), room_embeddings)
        similarity_matrix[i] = similarity_scores.cpu().detach().numpy()

    indices_dict = {}
    for obj_ind, obj_name in enumerate(obj_names):
        obj_cat = re.sub(r"_\d+$", "", obj_name)
        if obj_cat not in indices_dict:
            indices_dict[obj_cat] = [obj_ind]
        else:
            indices_dict[obj_cat].append(obj_ind)

    # if len(indices_dict)!=268:
    #     objs = get_object_names()
    #     result = [x for x in objs if x not in list(indices_dict.keys())]
    #     import pdb; pdb.set_trace()

    # assert len(indices_dict) == 268, "Error: indices_dict length is not 268"
    similarity_matrix_110 = np.zeros((len(indices_dict), len(room_names)))
    count = 0
    for obj_name in indices_dict:
        ind = indices_dict[obj_name]
        # assert len(ind) == 5
        similarity_matrix_110[count] = similarity_matrix[ind].mean(axis=0)
        count += 1

    av_precision = []
    for index, obj in enumerate(list(indices_dict.keys())):
        obj_cat = re.sub(r"_\d+$", "", obj)
        y_scores = similarity_matrix_110[index, :]
        y_true_in = [0]*len(room_names)
        for i, room in enumerate(room_names):
            assigned = False
            if room not in relationships.keys():
                continue
            if obj_cat in relationships[room] or obj_cat == room:
                y_true_in[i] = 1
                assigned = True
                break
        if not assigned:
            import pdb
            pdb.set_trace()

        ap = average_precision_score(y_true_in, y_scores, average='weighted')
        # precision, recall, thresholds = precision_recall_curve(y_true_in, y_scores)
        # import pdb; pdb.set_trace()
        av_precision.append(ap)
        # print(room, '\t', ap)
        # print('\n\n')
    mAP = sum(av_precision)/len(av_precision)
    # print(f"mAP: {mAP}")
    return mAP


def calculate_statistics(obj_embeddings, room_embeddings, obj_names, room_names, relationships, filename='obj_room.txt'):
    similarity_matrix = np.zeros((len(obj_embeddings), len(room_embeddings)))
    for i, obj_emb in enumerate(obj_embeddings):
        similarity_scores = torch.nn.functional.cosine_similarity(
            obj_emb.unsqueeze(0), room_embeddings)
        similarity_matrix[i] = similarity_scores.cpu().detach().numpy()

    indices_dict = {}
    for obj_ind, obj_name in enumerate(obj_names):
        obj_cat = re.sub(r"_\d+$", "", obj_name)
        if obj_cat not in indices_dict:
            indices_dict[obj_cat] = [obj_ind]
        else:
            indices_dict[obj_cat].append(obj_ind)

    similarity_matrix_110 = np.zeros((len(indices_dict), len(room_names)))
    count = 0
    for obj_name in indices_dict:
        ind = indices_dict[obj_name]
        # assert len(ind) == 5
        similarity_matrix_110[count] = similarity_matrix[ind].mean(axis=0)
        count += 1

    # Calculate the rank of each room assigned to each object
    ranks = {}
    predictions = np.zeros(len(indices_dict))
    y_true = np.zeros(len(indices_dict))
    hit = 0
    top_3_hit = 0
    top_5_hit = 0
    op_dict = {}
    for i, obj_name in enumerate(list(indices_dict.keys())):
        sorted_indices = np.argsort(similarity_matrix_110[i])[::-1]
        sorted_room_names = [room_names[index] for index in sorted_indices]
        op_dict[obj_name] = sorted_room_names
        assigned = False
        for room_ind, room in enumerate(room_names):
            # Regular expression to remove the image number from the obj name
            obj_cat = re.sub(r"_\d+$", "", obj_name)
            if obj_cat in relationships[room]:
                true_room_name = room
                y_true[i] = room_ind
                assigned = True
                break
        assert assigned == True

        # Calculate the rank of the true room among the sorted rooms
        true_room_rank = sorted_room_names.index(true_room_name) + 1
        # Save the rank for this object
        ranks[obj_name] = true_room_rank
        predictions[i] = sorted_indices[0]
        if true_room_name == sorted_room_names[0]:
            hit += 1
        if true_room_name in sorted_room_names[:3]:
            top_3_hit += 1
        if true_room_name in sorted_room_names[:5]:
            top_5_hit += 1

    with open(filename, 'w') as file:
        import pdb; pdb.set_trace()
        for key, value in op_dict.items():
            file.write(f"{key}: {', '.join(value)}\n")
    total_mrr = 0  # Mean Reciprocal Rank
    for rank in ranks.values():
        total_mrr += 1/rank
    mean_mrr = total_mrr/len(ranks)
    hit = hit/len(indices_dict)
    top_3_hit = top_3_hit/len(indices_dict)
    top_5_hit = top_5_hit/len(indices_dict)
    print("Mean reciprocal rank:", mean_mrr)
    print("Hit Ratio:", hit)
    print("Top 3 Hit Ratio:", top_3_hit)
    print("Top 5 Hit Ratio:", top_5_hit)

    labels = range(len(room_names))

    cm = confusion_matrix(y_true, predictions, labels=labels)
    from tabulate import tabulate
    table_data_final = []
    for i, room in enumerate(room_names):
        table_data = []
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        print(f"Confusion matrix for class {room}:")
        print(
            f"True Positive: {tp}, False Positive: {fp}, False Negative: {fn}, True Negative: {tn}")
        table_data.append([tp, fn])
        table_data.append([fp, tn])
        print(tabulate(table_data, tablefmt="grid"))
        table_data_final.append([room, tp, fp, fn, tn])

    headers = ["Label", "True Positive", "False Positive",
               "False Negative", "True Negative"]
    print(tabulate(table_data_final, headers=headers, tablefmt="grid"))

    precision = precision_score(y_true, predictions, average='weighted')
    recall = recall_score(y_true, predictions, average='weighted')
    print(
        f"Weighted Average Values \n Precision: {precision}\n Recall: {recall}")

    # av_precision = []
    # for index, room in enumerate(room_names):
    #     y_scores = similarity_matrix[:,index]
    #     y_true_in = (y_true==index).astype(int)
    #     ap = average_precision_score(y_true_in, y_scores, average='weighted')
    #     precision, recall, thresholds = precision_recall_curve(y_true_in, y_scores)
    #     av_precision.append(ap)
    #     print(room, '\t', ap)
    #     print('\n\n')

    # mAP = sum(av_precision)/len(av_precision)

    # print("Mean Average Precision = ", mAP)
    report = classification_report(y_true, predictions, labels=labels)
    print('Classification Report:')
    print(report)
    return hit, top_3_hit, top_5_hit
