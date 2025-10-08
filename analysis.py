from read_data import read_tabfact, read_scitab, qt_summ_load, scigen_load, read_pubhealth
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import ast
import statistics

def analyze_tabfact():
    train_tabfact = read_tabfact("/home/sonlt/drive/data/tabfact", type="train")
    val_tabfact = read_tabfact("/home/sonlt/drive/data/tabfact", type="val")
    test_tabfact = read_tabfact("/home/sonlt/drive/data/tabfact", type="test")

    label_train = []
    for t in train_tabfact:
        label_train.append(t[2])

    label_val= []
    for t in val_tabfact:
        label_val.append(t[2])

    label_test= []
    for t in test_tabfact:
        label_test.append(t[2])

    print(Counter(label_train))
    print(Counter(label_val))
    print(Counter(label_test))


def analyze_scitab():
    data = read_scitab("/home/sonlt/drive/data/scitab")
    label = []
    for t in data:
        label.append(t['label'])
    
    for t in data:
        pass
    print(Counter(label))

# def analyze_feverous():
#     train_data = read_feverous("/home/sonlt/drive/data/feverous", type="train")
#     dev_data = read_feverous("/home/sonlt/drive/data/feverous", type="dev")

#     label_train = []
#     num_evidence_train = []
#     for t in train_data:
#         label_train.append(t["label"])
#         num_evidence_train.append(len(t['evidence']))

#     label_val= []
#     num_evidence_val = []
#     for t in dev_data:
#         label_val.append(t["label"])
#         num_evidence_val.append(len(t['evidence']))

#     # print(Counter(label_train))
#     # print(Counter(label_val))

#     print(sum(num_evidence_train))
#     print(sum(num_evidence_val))

#     train_plot = sns.displot(num_evidence_train)
#     plt.savefig('train.png') 

#     val_plot = sns.displot(num_evidence_val)
#     plt.savefig('val.png') 


def change_table(col_lst, content_lst):
    df = {}
    for i in range(0, len(col_lst)):
        key = col_lst[i]
        val_list = []
        for j in range(0, len(content_lst)):
            val_list.append(content_lst[j][i])
        df[key] = val_list
        
    return pd.DataFrame(df)


def analyze_qtsumm():
    def qt_parse(data):
        new_data = []
        for d in data:
            table = change_table(d['table']['header'], d['table']['rows'])
            d['table_df'] = table
            new_data.append(d)

        return new_data
    
    def analyze_table_size(data):
        c_t = []
        r_t = []
        s_t = []
        for t in data:
            c_t.append(t['table_df'].shape[1])
            r_t.append(t['table_df'].shape[0])
            s_t.append(t['table_df'].size)

        return c_t, r_t, s_t
    
    def analyze_text_length(data):
        length_tokens = []
        for t in data:
            length_tokens.append(len(t['summary'].split()))
        
        return length_tokens
    
    qt_train, qt_dev, qt_test = qt_summ_load()

    train = qt_parse(qt_train)
    dev = qt_parse(qt_dev)
    test = qt_parse(qt_test)

    c_t, r_t, s_t = analyze_table_size(train)
    print(max(c_t))
    print(max(r_t))
    print(min(c_t))
    print(min(r_t))
    print(statistics.mode(c_t))
    print(statistics.mode(r_t))
    print("-----")

    c_t, r_t, s_t = analyze_table_size(dev)
    print(max(c_t))
    print(max(r_t))
    print(min(c_t))
    print(min(r_t))
    print(statistics.mode(c_t))
    print(statistics.mode(r_t))
    print("-----")

    c_t, r_t, s_t = analyze_table_size(test)
    print(max(c_t))
    print(max(r_t))
    print(min(c_t))
    print(min(r_t))
    print(statistics.mode(c_t))
    print(statistics.mode(r_t))
    print("-----------")
    print("-----------")
    
    lo = analyze_text_length(train)
    print(max(lo))
    print(min(lo))
    print(statistics.mean(lo))
    print("-----------")
    lo = analyze_text_length(dev)
    print(max(lo))
    print(min(lo))
    print(statistics.mean(lo))
    print("-----------")
    lo = analyze_text_length(test)
    print(max(lo))
    print(min(lo))
    print(statistics.mean(lo))
    print("-----------")


def analyze_scigen():
    def sc_parse(data):
        new_data = []
        for d in data:
            table = change_table(ast.literal_eval(d['table_column_names']), ast.literal_eval(d['table_content_values']))
            d['table_df'] = table
            new_data.append(d)

        return new_data
    
    def analyze_table_size(data):
        c_t = []
        r_t = []
        s_t = []
        for t in data:
            c_t.append(t['table_df'].shape[1])
            r_t.append(t['table_df'].shape[0])
            s_t.append(t['table_df'].size)

        return c_t, r_t, s_t
    
    def analyze_text_length(data):
        length_tokens = []
        for t in data:
            length_tokens.append(len(t['text'].split()))
        
        return length_tokens
    
    qt_train, qt_dev, qt_test = scigen_load()

    train = sc_parse(qt_train)
    dev = sc_parse(qt_dev)
    test = sc_parse(qt_test)

    c_t, r_t, s_t = analyze_table_size(train)
    print(max(c_t))
    print(max(r_t))
    print(min(c_t))
    print(min(r_t))
    print(statistics.mode(c_t))
    print(statistics.mode(r_t))
    print("-----")

    c_t, r_t, s_t = analyze_table_size(dev)
    print(max(c_t))
    print(max(r_t))
    print(min(c_t))
    print(min(r_t))
    print(statistics.mode(c_t))
    print(statistics.mode(r_t))
    print("-----")

    c_t, r_t, s_t = analyze_table_size(test)
    print(max(c_t))
    print(max(r_t))
    print(min(c_t))
    print(min(r_t))
    print(statistics.mode(c_t))
    print(statistics.mode(r_t))
    print("-----")
    print("-----------")
    
    lo = analyze_text_length(train)
    print(max(lo))
    print(min(lo))
    print(statistics.mean(lo))
    print("-----------")
    lo = analyze_text_length(dev)
    print(max(lo))
    print(min(lo))
    print(statistics.mean(lo))
    print("-----------")
    lo = analyze_text_length(test)
    print(max(lo))
    print(min(lo))
    print(statistics.mean(lo))
    print("-----------")


def analyze_scitab():
    def analyze_table_size(data):
        c_t = []
        r_t = []
        s_t = []
        for t in data:
            c_t.append(t['table'].shape[1])
            r_t.append(t['table'].shape[0])
            s_t.append(t['table'].size)

        return c_t, r_t, s_t

    sci_tab = read_scitab('/home/s2320014/data/scitab')

    c_t, r_t, s_t = analyze_table_size(sci_tab)
    print(max(c_t))
    print(max(r_t))
    print(min(c_t))
    print(min(r_t))
    print(statistics.mode(c_t))
    print(statistics.mode(r_t))
    print(statistics.mean(c_t))
    print(statistics.mean(r_t))
    print("-----------")
    print("-----------")


def analyze_pubhealthtab():
    def analyze_table_size(data):
        c_t = []
        r_t = []
        s_t = []
        for t in data:
            c_t.append(t['table_df'].shape[1])
            r_t.append(t['table_df'].shape[0])
            s_t.append(t['table_df'].size)

        return c_t, r_t, s_t

    pubhealth_tab = read_pubhealth('/home/s2320014/data/pubhealthtab', type="test")

    c_t, r_t, s_t = analyze_table_size(pubhealth_tab)
    print(max(c_t))
    print(max(r_t))
    print(min(c_t))
    print(min(r_t))
    print(statistics.mode(c_t))
    print(statistics.mode(r_t))
    print(statistics.mean(c_t))
    print(statistics.mean(r_t))
    print("-----------")
    print("-----------")


if __name__ == '__main__':
    # analyze_qtsumm()
    # analyze_scigen()
    print("PubHealthTab")
    analyze_pubhealthtab()