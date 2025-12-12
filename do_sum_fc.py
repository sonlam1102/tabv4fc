from read_data import *
import torch
from transformers import TapexTokenizer, BartForConditionalGeneration, BartTokenizer


class TableFactSum:
    def __init__(self, model="tapex_large_100_new_cap"):
        self._tokenizer = TapexTokenizer.from_pretrained("./model/{}/encoder/".format(model))
        self._label_tokienizer = BartTokenizer.from_pretrained("./model/{}/decoder/".format(model))
        self._model = BartForConditionalGeneration.from_pretrained("./model/{}/model/".format(model))
        self._features = []
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._name = "Tab fact"

    def make_features(self, data: list):  # claim, table, label, title
        print("making features tab sum")
        self._features = []
        for d in tqdm(data):
            input = {}
            try:
                input['input'] = self._tokenizer(table=d[1].astype(str), query="Summarize the table based on the information from this claim: {}".format(d[0]),
                            padding="max_length", max_length=1000, 
                            truncation=True,
                            return_tensors="pt")
            except Exception as e:
                print(d[0])
                print(d[1].astype(str))
                raise e
            self._features.append(input)
    
    def generate_summary(self):
        print("Num sample {}: {}".format(self._name, len(self._features)))
        assert len(self._features) > 0
        self._model = self._model.to(self._device)
        self._model.eval()
        out_results = []
        for s in tqdm(self._features):
            inp = s["input"]
            inp['input_ids'] = inp['input_ids'].squeeze(1).to(self._device)
            inp['attention_mask'] = inp['attention_mask'].squeeze(1).to(self._device)
            out = self._model.generate(input_ids=inp['input_ids'],
                                attention_mask= inp['attention_mask'],
                                max_length=400, do_sample=True, temperature=0.2, num_beams=4)
            # out = self._model.generate(input_ids=inp['input_ids'],
            #                     attention_mask= inp['attention_mask'],
            #                     max_length=400, do_sample=False)
            out_results.append(self._label_tokienizer.batch_decode(out, skip_special_tokens=True)[0])
        return out_results


class SciTabSum(TableFactSum):
    def __init__(self, model="tapex_large_100_new_cap"):
        super().__init__(model)
        self._features = []
        self._name = "Sci Tab"
    
    def make_features(self, data: list):
        print("making features sci tab")
        self._features = []
        for d in tqdm(data):
            input = {}
            input['input'] = self._tokenizer(table=d['table'], query="Summarize the table based on the information from this claim: {}.".format(d['claim']),
                            padding="max_length", max_length=1000, 
                            truncation=True,
                            return_tensors="pt")
            self._features.append(input)


class PubHealthTabSum(TableFactSum):
    def __init__(self, model="tapex_large_100_new_cap"):
        super().__init__(model)
        self._features = []
        self._name = "PubHealth Tab"
    
    def make_features(self, data: list):
        print("making features pubhealth tab")
        self._features = []
        for d in tqdm(data):
            input = {}
            # print(d['table_df'])
            input['input'] = self._tokenizer(table=d['table_df'].astype(str), query="Summarize the table based on the information from this claim: {}".format(d['claim']),
                            padding="max_length", max_length=1000, 
                            truncation=True,
                            return_tensors="pt")
            self._features.append(input)


def make_evidece_tab_fact(data, out_sum):
    assert len(data) == len(out_sum)
    results = []
    for i in range(0, len(data)):
        results.append({
            "claim": data[i][0],
            "table": data[i][1].to_markdown(tablefmt="grid"),
            "evidence": out_sum[i],
            "label": "entailed" if data[i][2] == 1 else "refuted",
        })
    return results


def make_evidece_sci_tab(data, out_sum):
    assert len(data) == len(out_sum)
    results = []
    for i in range(0, len(data)):
        results.append({
            "claim": data[i]["claim"],
            "evidence": out_sum[i],
            "table": data[i]['table'].to_markdown(tablefmt="grid"),
            "label": data[i]["label"],
        })
    return results


def make_evidece_pubhealth_tab(data, out_sum):
    assert len(data) == len(out_sum)
    results = []
    for i in range(0, len(data)):
        results.append({
            "claim": data[i]["claim"],
            "evidence": out_sum[i],
            "table": data[i]['table_df'].to_markdown(tablefmt="grid"),
            "label": data[i]["label"],
        })
    return results



if __name__ == "__main__":
    ## MAKE SUM

    print("Make summary")
    tab_fact = read_tabfact('/home/data/tabfact', type="test")
    sum_model1 = TableFactSum(model="tapex_large_100")
    sum_model1.make_features(tab_fact)
    out_sum1 = sum_model1.generate_summary()
    tab_fact_sum = make_evidece_tab_fact(tab_fact, out_sum1)
    with open("./tab_fact_FC.json", "w", encoding='utf-8') as f:
        json.dump(tab_fact_sum, f, ensure_ascii=False, indent=4)
    f.close()

    sci_tab = read_scitab('/home/data/scitab')
    sum_model2 = SciTabSum(model="tapex_large_100")
    sum_model2.make_features(sci_tab)
    out_sum2 = sum_model2.generate_summary()
    sci_tab_sum = make_evidece_sci_tab(sci_tab, out_sum2)
    with open("./sci_tab_FC.json", "w", encoding='utf-8') as f:
        json.dump(sci_tab_sum, f, ensure_ascii=False, indent=4)
    f.close()

    pubhealth_tab = read_pubhealth("/home/data/pubhealthtab", type="test")
    sum_model3 = PubHealthTabSum(model="tapex_large_100")
    sum_model3.make_features(pubhealth_tab)
    out_sum3 = sum_model3.generate_summary()
    pubhealth_tab_sum = make_evidece_pubhealth_tab(pubhealth_tab, out_sum3)
    with open("./pubhealth_tab_FC.json", "w", encoding='utf-8') as f:
        json.dump(pubhealth_tab_sum, f, ensure_ascii=False, indent=4)
    f.close()
