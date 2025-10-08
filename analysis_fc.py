import json 
import statistics
import seaborn as snb
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report
from llms import retrieve_verification_results

def analyze_data(data):
    label = []
    sum_len = []
    claim_len = []
    for d in data:
        label.append(d['label'])
        sum_len.append(len(d['evidence'].split()))
        claim_len.append(len(d['claim'].split()))
    
    return label, sum_len, claim_len


if __name__ == "__main__":
    # with open("./pubhealth_tab_FC(1).json", "r") as f:
    # # with open("./tab_fact_FC(1).json", "r") as f:
    # # with open("./sci_tab_FC(1).json", "r") as f:
    #     data = json.load(f)
    # f.close()

    # lb_data, sum_len, claim_len = analyze_data(data)
    # # print(len(lb_data))
    # # print(lb_data.count("supports"))
    # # print(lb_data.count("not enough info"))
    # # print(lb_data.count("refutes"))


    # print(max(sum_len))
    # print(min(sum_len))
    # print(statistics.mean(sum_len))

    # sns_plot = snb.displot(x=sum_len, bins=10)
    # # sns_plot = snb.countplot()
    # fig = sns_plot.figure
    # fig.savefig("sum_len_pubhealthtab.png")
    # plt.show()

    print("PubHealthTab---")
    with open("./results/pubhealth_tab_FC_result-qwen25.json", "r") as f:
        result = json.load(f)
    f.close()
    g, p, new_results = retrieve_verification_results(result, label=3, cased=True)
    label_map = {
        2: "SUPPORTS",
        1: "NEI",
        0: "REFUTES"
    }
    print("Test result micro: {}\n".format(f1_score(g, p, average='micro')))
    print("Test result macro: {}\n".format(f1_score(g, p, average='macro')))
    g = [label_map[i] for i in g]
    p = [label_map[i] for i in p]
    cm = confusion_matrix(g, p, labels=["REFUTES", "NEI", "SUPPORTS"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["REFUTES", "NEI", "SUPPORTS"])
    disp.plot(cmap=plt.cm.Greys)
    # plt.savefig("pubhealthtab_cm.png")
    print(classification_report(g, p, target_names=["REFUTES", "NEI", "SUPPORTS"], digits=4))
    print("----------------------")

    print("SCITAB---")
    with open("./results/sci_tab_FC_result-qwen25.json", "r") as f:
        result_pubhealthtab = json.load(f)
    f.close()
    g, p, new_results = retrieve_verification_results(result_pubhealthtab, label=3, cased=False)
    label_map = {
        2: "supports",
        1: "nei",
        0: "refutes"
    }
    print("Test result micro: {}\n".format(f1_score(g, p, average='micro')))
    print("Test result macro: {}\n".format(f1_score(g, p, average='macro')))
    g = [label_map[i] for i in g]
    p = [label_map[i] for i in p]
    cm = confusion_matrix(g, p, labels=["refutes", "nei", "supports"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["refutes", "nei", "supports"])
    disp.plot(cmap=plt.cm.Greys)
    # plt.savefig("scitab_cm.png")
    print(classification_report(g, p, target_names=["refutes", "nei", "supports"], digits=4))
    print("----------------------")

    print("TabFact---")
    with open("./results/tab_fact_FC_result-qwen25.json", "r") as f:
        result_pubhealthtab = json.load(f)
    f.close()
    g, p, new_results = retrieve_verification_results(result_pubhealthtab, label=2, cased=False)
    label_map = {
        1: "entailed",
        0: "refuted"
    }
    print("Test result micro: {}\n".format(f1_score(g, p, average='micro')))
    print("Test result macro: {}\n".format(f1_score(g, p, average='macro')))
    g = [label_map[i] for i in g]
    p = [label_map[i] for i in p]

    cm = confusion_matrix(g, p, labels=["refuted", "entailed"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["refuted", "entailed"])
    disp.plot(cmap=plt.cm.Greys)
    # plt.savefig("tabfact_cm.png")
    print(classification_report(g, p, target_names=["refuted", "entailed"], digits=4))
    print("----------------------")
