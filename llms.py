from read_data import *
import torch
from transformers import TapexTokenizer, BartForConditionalGeneration, BartTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


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


def make_verification_prompt(claim, table, description, label=2, has_summary=True, has_table=True):
    # Summarization of table: {description}
    temp = True
    if label == 2:
        if has_summary and has_table:
            prompt = f"""
            You are an assistant that help to verify the claim. 
            The claim is: {claim}
            The table that containing the information for verifying the claim:
                {table}
            
            Summarization of table: {description}

            Based on the table and the summarization, please think and determine the truthfulness of the claim. The truthfulness must be one of these values: entailed or refuted.
            <RESPONSE>: 
            """
        else:
            if has_table:
                prompt = f"""
                You are an assistant that help to verify the claim. 
                The claim is: {claim}
                The table that containing the information for verifying the claim:
                    {table}

                Based on the table, please think and determine the truthfulness of the claim. The truthfulness must be one of these values: entailed or refuted.
                <RESPONSE>: 
                """
            elif has_summary:
                prompt = f"""
                You are an assistant that help to verify the claim. 
                The claim is: {claim}
                Evidence summarized from the table: {description}

                Based on the evidence, please think and determine the truthfulness of the claim. The truthfulness must be one of these values: entailed or refuted.
                <RESPONSE>: 
                """
            else:
                prompt = f"""
                You are an assistant that help to verify the claim. 
                The claim is: {claim}

                Please think and determine the truthfulness of the claim. The truthfulness must be one of these values: entailed or refuted.
                <RESPONSE>: 
                """

    else:
        if has_summary and has_table:
            prompt = f"""
            You are an assistant that help to verify the claim. 
            The claim is: {claim}
            The table that containing the information for verifying the claim:
                {table}
            Summarization of table: {description}

            Based on the table and the summarization, please think and determine the truthfulness of the claim. The truthfulness must be one of these values: supported, refuted or not enough information. 
            <RESPONSE>: 
            """
        else:
            if has_table:
                prompt = f"""
                You are an assistant that help to verify the claim. 
                The claim is: {claim}
                The table that containing the information for verifying the claim:
                    {table}

                Based on the table, please think and determine the truthfulness of the claim. The truthfulness must be one of these values: supported, refuted or not enough information. 
                <RESPONSE>: 
                """
            elif has_summary:
                prompt = f"""
                You are an assistant that help to verify the claim. 
                The claim is: {claim}
                Evidence summarized from the table: {description}

                Based on the evidence, please think and determine the truthfulness of the claim. The truthfulness must be one of these values: supported, refuted or not enough information. 
                <RESPONSE>: 
                """
            else:
                prompt = f"""
                You are an assistant that help to verify the claim. 
                The claim is: {claim}

                Please think and determine the truthfulness of the claim. The truthfulness must be one of these values: supported, refuted or not enough information. 
                <RESPONSE>: 
                """
    # print(prompt)
    # raise Exception
    return prompt


def load_peft_model_text(peft_model_name, device="auto", quantile=True, flash_attention=True):
    processor = AutoTokenizer.from_pretrained(
        peft_model_name,
        padding_side="left",
        truncation_side="left",
        token=""
    )

    quantization_config = BitsAndBytesConfig(
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        load_in_4bit=True,
        load_in_8bit=False,
    )

    if quantile:
        model = AutoModelForCausalLM.from_pretrained(
        peft_model_name,
        quantization_config=quantization_config,
        token="",
        device_map=device,
        use_flash_attention_2=flash_attention
    )
    else:
        model = AutoModelForCausalLM.from_pretrained(
        peft_model_name,
        token="",
        device_map=device,
        use_flash_attention_2=flash_attention
    )

    return processor, model


@torch.inference_mode()
def do_inference_text(model, processor, prompt, new_token=10):
    inputs = processor(prompt, return_tensors="pt").to(model.device)
    model.generation_config.pad_token_id = processor.pad_token_id

    output_ids = model.generate(
        **inputs,
        max_new_tokens=new_token,
        do_sample=False,
        pad_token_id=processor.eos_token_id
    )
    return processor.decode(output_ids[0])


def retrieve_verification_results(data, label=2, cased=False):
    def filter_results2labels(response):
        response = response.split("<RESPONSE>")[-1]
        if ("entailed" in response or "Entailed" in response):
            return "entailed"
        else:
            return "refuted"
        
    def filter_results3labels(response):
        response = response.split("<RESPONSE>")[-1]
        if ("supported" in response or "Supported" in response or "SUPPORTED" in response)  and "not supported" not in response:
            return "supports" if not cased else "SUPPORTS"
        elif ("refuted" in response or "Refuted" in response or "REFUTED" in response) or "not supported" in response:
            return "refutes" if not cased else "REFUTES"
        else:
            return "not enough info" if not cased else "NOT ENOUGH INFO"

    if label == 2:
        label2inx = {
            "entailed": 1,
            "refuted": 0
        }
    else:
        if cased == True:
            label2inx = {
            "SUPPORTS": 2,
            "NOT ENOUGH INFO": 1,
            "REFUTES": 0
        }
        else:
            label2inx = {
            "supports": 2,
            "not enough info": 1,
            "refutes": 0
        }

    ground_truth = []
    predict = []
    for d in data:
        if label == 2:
            predict.append(label2inx[filter_results2labels(d['results'])])
            ground_truth.append(label2inx[d['label']])
            d['predict'] = filter_results2labels(d['results'])
        else:
            predict.append(label2inx[filter_results3labels(d['results'])])
            ground_truth.append(label2inx[d['label']])
            d['predict'] = filter_results3labels(d['results'])
    
    return ground_truth, predict, data


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


def create_verification_prompt(dataset, model, processor, new_token=10, label=2, has_summary=True, has_table=True):
    results = []
    temp = True
    print("---performing verification .....----")
    for sample in tqdm(dataset):
        prompt = make_verification_prompt(sample['claim'], sample['table'], sample['evidence'], label=label, has_summary=has_summary, has_table=has_table)
        if temp:
            print(prompt)
            temp = False

        try:
            results.append({
                **sample,
                'results': do_inference_text(model, processor, prompt, new_token)
            })
            # print(results)
            # raise Exception
        except Exception as e:
            # raise e
            print(e)
            print(sample['claim_id'])
            results.append({
                **sample,
                'results': "This claim is supported"
            })
    return results


if __name__ == "__main__":
    # ## MAKE SUM

    # print("Make summary")
    # tab_fact = read_tabfact('/home/sonlt/drive/data/tabfact', type="test")
    # sum_model1 = TableFactSum(model="tapex_large_100")
    # sum_model1.make_features(tab_fact)
    # out_sum1 = sum_model1.generate_summary()
    # tab_fact_sum = make_evidece_tab_fact(tab_fact, out_sum1)
    # with open("./tab_fact_FC.json", "w", encoding='utf-8') as f:
    #     json.dump(tab_fact_sum, f, ensure_ascii=False, indent=4)
    # f.close()

    # sci_tab = read_scitab('/home/sonlt/drive/data/data/scitab')
    # sum_model2 = SciTabSum(model="tapex_large_100")
    # sum_model2.make_features(sci_tab)
    # out_sum2 = sum_model2.generate_summary()
    # sci_tab_sum = make_evidece_sci_tab(sci_tab, out_sum2)
    # with open("./sci_tab_FC.json", "w", encoding='utf-8') as f:
    #     json.dump(sci_tab_sum, f, ensure_ascii=False, indent=4)
    # f.close()

    # pubhealth_tab = read_pubhealth("/home/sonlt/drive/data/data/pubhealthtab", type="test")
    # sum_model3 = PubHealthTabSum(model="tapex_large_100")
    # sum_model3.make_features(pubhealth_tab)
    # out_sum3 = sum_model3.generate_summary()
    # pubhealth_tab_sum = make_evidece_pubhealth_tab(pubhealth_tab, out_sum3)
    # with open("./pubhealth_tab_FC.json", "w", encoding='utf-8') as f:
    #     json.dump(pubhealth_tab_sum, f, ensure_ascii=False, indent=4)
    # f.close()
    
    ## RUN LLMs
    print("==== Running LLMs ====")
    # processor, model = load_peft_model_text("meta-llama/Llama-3.3-70B-Instruct", flash_attention=True)
    # processor, model = load_peft_model_text("mistralai/Mixtral-8x7B-Instruct-v0.1", flash_attention=False, device="cuda:1")
    
    # processor, model = load_peft_model_text("meta-llama/Llama-3.3-70B-Instruct", flash_attention=False, device="cuda:1")
    # processor, model = load_peft_model_text("mistralai/Mixtral-8x7B-Instruct-v0.1")
    # processor, model = load_peft_model_text("/data/huggingface_models/Qwen2.5-72B-Instruct")
    # processor, model = load_peft_model_text("/data/huggingface_models/Qwen2.5-14B-Instruct")
    # processor, model = load_peft_model_text("Qwen/Qwen2.5-14B-Instruct")
    # processor, model = load_peft_model_text("Qwen/Qwen2.5-72B-Instruct")
    processor, model = load_peft_model_text("Qwen/Qwen2.5-32B-Instruct")

    # processor, model = load_peft_model_text("deepseek-ai/DeepSeek-R1-Distill-Llama-70B", flash_attention=True)
    # processor, model = load_peft_model_text("/data/huggingface_models/DeepSeek-R1-Distill-Llama-70B", flash_attention=True)
    # processor, model = load_peft_model_text("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", flash_attention=True)
    # processor, model = load_peft_model_text("/data/huggingface_models/DeepSeek-R1-Distill-Qwen-32B", flash_attention=True)

    num_label = 3
    cased = True
    has_summary = True
    has_table = True

    with open("./pubhealth_tab_FC(1).json", "r") as f:
    # with open("./tab_fact_FC(1).json", "r") as f:
    # with open("./sci_tab_FC(1).json", "r") as f:
        data = json.load(f)
    f.close()

    results = create_verification_prompt(data, model, processor, new_token=10, label=num_label, has_summary=has_summary, has_table=has_table)
    g, p, new_results = retrieve_verification_results(results, label=num_label, cased=cased)

    with open('./pubhealth_tab_FC_result-qwen32.json', 'w', encoding='utf-8') as f:
    # with open('./tab_fact_FC_result-qwen32.json', 'w', encoding='utf-8') as f:
    # with open('./sci_tab_FC_result-qwen32.json', 'w', encoding='utf-8') as f:
        json.dump(new_results, f, ensure_ascii=False, indent=4)
    f.close()

    print("Test result micro: {}\n".format(f1_score(g, p, average='micro')))
    print("Test result macro: {}\n".format(f1_score(g, p, average='macro')))
    print("Test result Accuracy: {}\n".format(accuracy_score(g, p)))
    print(confusion_matrix(g, p, labels=[0, 1])) if num_label == 2 else print(confusion_matrix(g, p, labels=[0, 1, 2]))

    # ## DEMO test 
    # num_label = 3
    # cased = False
    # with open("./sci_tab_FC_result-llama.json", "r") as f:
    #     results = json.load(f)
    # f.close()
    # g, p, new_results = retrieve_verification_results(results, label=num_label, cased=cased)
    # print("Test result micro: {}\n".format(f1_score(g, p, average='micro')))
    # print("Test result macro: {}\n".format(f1_score(g, p, average='macro')))
    # print("Test result Accuracy: {}\n".format(accuracy_score(g, p)))
    # print(confusion_matrix(g, p, labels=[0, 1])) if num_label == 2 else print(confusion_matrix(g, p, labels=[0, 1, 2]))

