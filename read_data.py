import pandas as pd
import json
from tqdm import tqdm
from datasets import load_dataset

def read_tabfact(path, type="train"):
    data = []
    with open(path+"/{}_examples.json".format(type), "r") as f:
        train = json.load(f)
    f.close()

    for key, val in tqdm(train.items()):
        path_to_csv = path + "/misc_data/all_csv/{}".format(key)
        table_data = pd.read_csv(path_to_csv, sep="#")
        lst_claim = val[0]
        lst_label = val[1]

        title = val[2]
        assert len(lst_claim) == len(lst_label)
        for i in range(0, len(lst_claim)):
            data.append((lst_claim[i], table_data, lst_label[i], title)) # claim, table, label, title

    return data


def read_scitab(path):
    def parse_table(col_lst, content_lst):
        df = {}
        for i in range(0, len(col_lst)):
            key = col_lst[i]
            val_list = []
            for j in range(0, len(content_lst)):
                val_list.append(content_lst[j][i])
            df[key] = val_list
        
        return pd.DataFrame(df)

    with open(path+"/sci_tab.json", "r") as f:
        dataset = json.load(f)
    f.close()

    for d in dataset:
        d['table'] = parse_table(d['table_column_names'], d['table_content_values'])

    return dataset

# def read_feverous(path, type="train"):
#     data = []
#     if type == "train":
#         with open(path+"/feverous_train_challenges.jsonl", "r") as f:
#             for line in f:
#                 data.append(json.loads(line))
#         f.close()
#     else:
#         with open(path+"/feverous_dev_challenges.jsonl", "r") as f:
#             for line in f:
#                 data.append(json.loads(line))
#         f.close()
    
#     return data

def read_pubhealth(path, type="train"):
    data = []
    if type == "train":
        with open(path+"/pubhealthtab_trainset.jsonl", "r") as f:
            for line in f:
                data.append(json.loads(line))
        f.close()
    elif type == "dev":
        with open(path+"/pubhealthtab_evalset.jsonl", "r") as f:
            for line in f:
                data.append(json.loads(line))
        f.close()
    else:
        with open(path+"/pubhealthtab_testset.jsonl", "r") as f:
            for line in f:
                data.append(json.loads(line))
        f.close()
    
    for d in data:
        tmp = pd.read_html(d['table']['html_code'], header=0)
        try:
            assert len(tmp) == 1
            d['table_df'] = tmp[-1]
            d['table_df'] = d['table_df'].dropna(axis=1, how='all')
            d['table_df'] = d['table_df'].dropna(axis=0, how='all')
        except Exception as e:
            print(d['_id'])
            d['table_df'] = tmp[-1]
            d['table_df'] = d['table_df'].dropna(axis=1, how='all')
            d['table_df'] = d['table_df'].dropna(axis=0, how='all')

    return data

def qt_summ_load():
    train = load_dataset("yale-nlp/QTSumm", split="train")
    dev = load_dataset("yale-nlp/QTSumm", split="validation")
    test = load_dataset("yale-nlp/QTSumm", split="test")

    return train, dev, test

def scigen_load():
    train = load_dataset("kasnerz/scigen", split="train")
    dev = load_dataset("kasnerz/scigen", split="validation")
    test = load_dataset("kasnerz/scigen", split="test")

    return train, dev, test


if __name__ == "__main__":
    # tf_train = read_scitab("/home/sonlt/drive/data/scitab")
    # print(len(tf_train))
    # print(tf_train[100]['table'])
    ph = read_pubhealth("/home/sonlt/drive/data/pubhealthtab")
    print(len(ph))
    print(ph[100]['table_df'])
    print(ph[100]['table']['html_code'])
    print(ph[100]['_id'])