from read_data import *
import torch
from transformers import TapexTokenizer, BartForConditionalGeneration, BartTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import argparse


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
    return prompt


def load_peft_model_text(peft_model_name, device="auto", quantile=True, flash_attention=True):
    processor = AutoTokenizer.from_pretrained(
        peft_model_name,
        padding_side="left",
        truncation_side="left",
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
        device_map=device,
        use_flash_attention_2=flash_attention
    )
    else:
        model = AutoModelForCausalLM.from_pretrained(
        peft_model_name,
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


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="/home/data/tab_fact_FC.json", type=str)
    parser.add_argument('--data_name', default="tabfact", type=str)
    parser.add_argument('--model', default="Qwen/Qwen2.5-72B-Instruct", type=str)
    parser.add_argument('--no_summary', default=False, action='store_true')
    parser.add_argument('--no_table', default=False, action='store_true')
    parser.add_argument('--quantile', default=False, action='store_true')
    parser.add_argument('--flash_attn', default=False, action='store_true')
    args = parser.parse_args()
    return args


if __name__ == "__main__":    
    ## RUN LLMs
    print("==== Running LLMs ====")
    args = parser_args()
    processor, model = load_peft_model_text(args.model, quantile=args.quantile, flash_attention=args.flash_attn)

    has_summary = not args.no_summary
    has_table = not args.no_table
    
    if args.data_name == "tabfact":
        num_label = 2
        cased = False
    elif args.data_name == "scitab":
        num_label = 3
        cased = False
    elif args.data_name == "pubhealthtab":
        num_label = 3
        cased = True
    else:
        raise "Not specific dataset"
        
    print(args.data_name)
    print(num_label)
    print(cased)
    print(has_summary)
    print(has_table)
    with open(args.data_path, "r") as f:
        data = json.load(f)
    f.close()

    results = create_verification_prompt(data, model, processor, new_token=10, label=num_label, has_summary=has_summary, has_table=has_table)
    g, p, new_results = retrieve_verification_results(results, label=num_label, cased=cased)

    with open('./{}_FC_result-{}.json'.format(args.data_name, args.model.split("/")[-1]), 'w', encoding='utf-8') as f:
        json.dump(new_results, f, ensure_ascii=False, indent=4)
    f.close()

    print("Test result micro: {}\n".format(f1_score(g, p, average='micro')))
    print("Test result macro: {}\n".format(f1_score(g, p, average='macro')))
    print("Test result Accuracy: {}\n".format(accuracy_score(g, p)))
    print(confusion_matrix(g, p, labels=[0, 1])) if num_label == 2 else print(confusion_matrix(g, p, labels=[0, 1, 2]))