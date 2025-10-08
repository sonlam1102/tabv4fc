import copy
from datasets import load_dataset
import pandas as pd
import ast
from transformers import TapexTokenizer, BartForConditionalGeneration, BartTokenizer
from torch.utils.data import DataLoader
from transformers import AdamW
import torch
from tqdm import tqdm, trange 
from read_data import qt_summ_load, scigen_load
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
import bert_score
from adopt import ADOPT
import argparse

from rouge_score import rouge_scorer


# def qt_summ_load():
#     train = load_dataset("yale-nlp/QTSumm", split="train")
#     dev = load_dataset("yale-nlp/QTSumm", split="validation")
#     test = load_dataset("yale-nlp/QTSumm", split="test")

#     return train, dev, test

# def scigen_load():
#     train = load_dataset("kasnerz/scigen", split="train")
#     dev = load_dataset("kasnerz/scigen", split="validation")
#     test = load_dataset("kasnerz/scigen", split="test")

#     return train, dev, test

def change_table(col_lst, content_lst):
    df = {}
    for i in range(0, len(col_lst)):
        key = col_lst[i]
        val_list = []
        for j in range(0, len(content_lst)):
            val_list.append(content_lst[j][i])
        df[key] = val_list
        
    return pd.DataFrame(df)

def qt_parse(data):
    new_data = []
    for d in data:
        table = change_table(d['table']['header'], d['table']['rows'])
        d['table_df'] = table
        d['text'] = d['summary']
        d['query'] = d['query']
        new_data.append(d)

    return new_data


def sc_parse(data):
    new_data = []
    for d in data:
        table = change_table(ast.literal_eval(d['table_column_names']), ast.literal_eval(d['table_content_values']))
        d['table_df'] = table
        d['query'] = "Let summarize the table based on this caption: " + d['table_caption']
        new_data.append(d)

    return new_data


def make_features(tokenizer, label_tokienizer, data):
    print("making features")
    features = []
    for d in tqdm(data):
        input = {}
        input['input'] = tokenizer(table=d['table_df'], query=d['query'],
                        padding="max_length", max_length=1000, 
                        truncation=True,
                        return_tensors="pt")
        input['label'] = label_tokienizer(d['text'], padding="max_length", max_length=1000, 
                        truncation=True,
                        return_tensors="pt")
        # input['label'] = tokenizer(table=d['table_df'], answer=d['summary'], padding="max_length",max_length=1000, 
        #                   truncation=True,
        #                   return_tensors="pt")
        features.append(input)
    return features


def fine_tune_tapex(tokenizer, label_tokienizer, model, train, epoch=1, batch_size=4):
    train = make_features(tokenizer, label_tokienizer, train)
    train_data = DataLoader(train, batch_size=batch_size)
    # optimizer = AdamW(model.parameters(), lr=1e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(device)
    model = model.to(device)
    optimizer = ADOPT(model.parameters(), lr=1e-5)

    for e in trange(0, epoch):
        print("Epoch:", e)
        total_loss = 0
        model.train()
        for b in tqdm(train_data):
            optimizer.zero_grad()
            inp = b["input"]
            inp['input_ids'] = inp['input_ids'].squeeze(1).to(device)
            inp['attention_mask'] = inp['attention_mask'].squeeze(1).to(device)
            
            lab = b['label']['input_ids'].squeeze(1).to(device)
            outputs = model(input_ids=inp['input_ids'], attention_mask=inp['attention_mask'], labels=lab)
            loss = outputs.loss
            total_loss = total_loss + loss
            loss.backward()
            optimizer.step()
        print("Loss: {}".format(total_loss))

        # pt, gt = predict(tokenizer, label_tokienizer, model, dev)

        # print(compute_bleu(gt, pt))

    model.save_pretrained("./model/tapex_large_{}/model".format(str(epoch)), from_pt=True)
    tokenizer.save_pretrained("./model/tapex_large_{}/encoder/".format(str(epoch)))
    label_tokienizer.save_pretrained("./model/tapex_large_{}/decoder/".format(str(epoch)))

    print("Finished. Saved model.")
    return model, tokenizer, label_tokienizer


def eval_tapex(tokenizer, label_tokienizer, model, data):
    dev = make_features(tokenizer, label_tokienizer, data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()
    print("Predicting")
    out_results = []
    ground = []

    assert len(dev) == len(data)
    for s in data:
        ground.append(s['text'])
    for s in tqdm(dev):
        inp = s["input"]
        inp['input_ids'] = inp['input_ids'].squeeze(1).to(device)
        inp['attention_mask'] = inp['attention_mask'].squeeze(1).to(device)
        # out = model.generate(input_ids=inp['input_ids'],
        #                         attention_mask= inp['attention_mask'],
        #                         max_length=400, do_sample=True, temperature=0.2, num_beams=5)
        out = model.generate(input_ids=inp['input_ids'],
                                attention_mask= inp['attention_mask'],
                                max_length=400, do_sample=False)
        out_results.append(label_tokienizer.batch_decode(out, skip_special_tokens=True)[0])
    
    # print(out_results[0])
    # print(ground[0])
    assert len(ground) == len(out_results)
    return out_results, ground


def compute_bleu(grounds, preds):
    g_t = copy.deepcopy(grounds)
    p_t = copy.deepcopy(preds)
    score = 0
    for i in range(len(g_t)):
        if g_t[i] == '':
            g_t[i] = '<empty>'
        if p_t[i] == '':
            p_t[i] = '<empty>'
        score += sentence_bleu([word_tokenize(g_t[i])],
                               word_tokenize(p_t[i]),
                               smoothing_function=SmoothingFunction().method3
                               )

    score /= len(grounds)
    return score


def compute_rouge(grounds, preds):
    g_t = copy.deepcopy(grounds)
    p_t = copy.deepcopy(preds)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    score = 0
    for i in range(len(g_t)):
        if g_t[i] == '':
            g_t[i] = '<empty>'
        if p_t[i] == '':
            p_t[i] = '<empty>'

        temp_score = scorer.score(g_t[i], p_t[i])
        precision, recall, fmeasure = temp_score['rougeL']
        score = score + fmeasure

    score /= len(grounds)
    return score


def compute_meteor(grounds, preds):
    g_t = copy.deepcopy(grounds)
    p_t = copy.deepcopy(preds)
    score = 0
    for i in range(len(g_t)):
        if g_t[i] == '':
            g_t[i] = '<empty>'
        if p_t[i] == '':
            p_t[i] = '<empty>'
        score += meteor_score([word_tokenize(g_t[i])],
                               word_tokenize(p_t[i]))

    score /= len(grounds)
    return score


def compute_bertscore(grounds, preds):
    g_t = copy.deepcopy(grounds)
    p_t = copy.deepcopy(preds)
    score = 0
    for i in range(len(g_t)):
        if g_t[i] == '':
            g_t[i] = '<empty>'
        if p_t[i] == '':
            p_t[i] = '<empty>'

    precision, recall, fmeasure = bert_score.score(p_t, g_t, lang="en", verbose=False)
    return fmeasure.mean().item()


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--path', default="./model", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parser_args()
    qt_train, qt_dev, qt_test = qt_summ_load()
    sc_train, sc_dev, sc_test = scigen_load()


    qt_train = qt_parse(qt_train)
    qt_dev = qt_parse(qt_dev)
    qt_test = qt_parse(qt_test)

    sc_train = sc_parse(sc_train)
    sc_dev = sc_parse(sc_dev)
    sc_test = sc_parse(sc_test)

    # print(sc_dev[2])
    # print(qt_test[4])

    combined_train = qt_train + sc_train
    print(len(combined_train))
    # new_sc = sc_parse(sc_train)
    # print(new_sc[0]['table_df'])

    # fe = make_features(tokenizer, new_qt)
    # data = DataLoader(fe, batch_size=8)
    # print(data)


    # ## ERROR ANALYSIS
    # tokenizer = TapexTokenizer.from_pretrained("./model/tapex_large_200/encoder/")
    # label_tokienizer = BartTokenizer.from_pretrained("./model/tapex_large_200/decoder/")
    # model = BartForConditionalGeneration.from_pretrained("./model/tapex_large_200/model/")

    # sample_pred1, sample_ground1 = eval_tapex(tokenizer, label_tokienizer, model, qt_dev[0:5])
    # sample_pred2, sample_ground2 = eval_tapex(tokenizer, label_tokienizer, model, sc_dev[0:5])

    # for i in range(0, 5):
    #     qt_dev[i]['table_df'].to_csv("./samples/qt_sum_table_{}.csv".format(i))
    #     sc_dev[i]['table_df'].to_csv("./samples/scigen_table_{}.csv".format(i))

    # sample1 = pd.DataFrame({
    #     "pred": sample_pred1,
    #     "ground": sample_ground1
    # })
    # sample1.to_csv("./samples/qt_sum_pred.csv")

    # sample2 = pd.DataFrame({
    #     "pred": sample_pred2,
    #     "ground": sample_ground2
    # })
    # sample2.to_csv("./samples/scigen_pred.csv")


    if args.test == False:
        print("FINETUNING")
        tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large")
        label_tokienizer = BartTokenizer.from_pretrained("facebook/bart-large")
        model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-large")

        EPOCH = args.epoch
        BATCH_SIZE = args.batch_size
        print("Num epoch: {}".format(EPOCH))
        finetuned_model, f_tokenizer, f_label_tokenizer = fine_tune_tapex(tokenizer, label_tokienizer, model, combined_train, epoch=EPOCH, batch_size=BATCH_SIZE)

        print("-----Dev-----")
        dev_qt_pred, dev_qt_ground = eval_tapex(f_tokenizer, f_label_tokenizer, finetuned_model, qt_dev)
        dev_sci_pred, dev_sci_ground = eval_tapex(f_tokenizer, f_label_tokenizer, finetuned_model, sc_dev)
        print("BLEU QTSumm: {}".format(compute_bleu(dev_qt_ground, dev_qt_pred)))
        print("BLEU SciGEN: {}".format(compute_bleu(dev_sci_ground, dev_sci_pred)))
        print("ROUGE QTSumm: {}".format(compute_rouge(dev_qt_ground, dev_qt_pred)))
        print("ROUGE SciGEN: {}".format(compute_rouge(dev_sci_ground, dev_sci_pred)))
        print("METEOR QTSumm: {}".format(compute_meteor(dev_qt_ground, dev_qt_pred)))
        print("METEOR SciGEN: {}".format(compute_meteor(dev_sci_ground, dev_sci_pred)))
        print("BERTSCORE QTSumm: {}".format(compute_bertscore(dev_qt_ground, dev_qt_pred)))
        print("BERTSCORE SciGEN: {}".format(compute_bertscore(dev_sci_ground, dev_sci_pred)))


        print("-----Test-----")
        test_qt_pred, test_qt_ground = eval_tapex(f_tokenizer, f_label_tokenizer, finetuned_model, qt_test)
        test_sci_pred, test_sci_ground = eval_tapex(f_tokenizer, f_label_tokenizer, finetuned_model, sc_test)
        print("BLEU QTSumm: {}".format(compute_bleu(test_qt_ground, test_qt_pred)))
        print("BLEU SciGEN: {}".format(compute_bleu(test_sci_ground, test_sci_pred)))
        print("ROUGE QTSumm: {}".format(compute_rouge(test_qt_ground, test_qt_pred)))
        print("ROUGE SciGEN: {}".format(compute_rouge(test_sci_ground, test_sci_pred)))
        print("METEOR QTSumm: {}".format(compute_meteor(test_qt_ground, test_qt_pred)))
        print("METEOR SciGEN: {}".format(compute_meteor(test_sci_ground, test_sci_pred)))
        print("BERTSCORE QTSumm: {}".format(compute_bertscore(test_qt_ground, test_qt_pred)))
        print("BERTSCORE SciGEN: {}".format(compute_bertscore(test_sci_ground, test_sci_pred)))
    else:
        print("EVALUATING")
        tokenizer = TapexTokenizer.from_pretrained("{}/encoder/".format(args.path))
        label_tokienizer = BartTokenizer.from_pretrained("{}/decoder/".format(args.path))
        model = BartForConditionalGeneration.from_pretrained("{}/model/".format(args.path))

        print("-----Dev-----")
        dev_qt_pred, dev_qt_ground = eval_tapex(tokenizer, label_tokienizer, model, qt_dev)
        dev_sci_pred, dev_sci_ground = eval_tapex(tokenizer, label_tokienizer, model, sc_dev)
        print("BLEU QTSumm: {}".format(compute_bleu(dev_qt_ground, dev_qt_pred)))
        print("BLEU SciGEN: {}".format(compute_bleu(dev_sci_ground, dev_sci_pred)))
        print("ROUGE QTSumm: {}".format(compute_rouge(dev_qt_ground, dev_qt_pred)))
        print("ROUGE SciGEN: {}".format(compute_rouge(dev_sci_ground, dev_sci_pred)))
        print("METEOR QTSumm: {}".format(compute_meteor(dev_qt_ground, dev_qt_pred)))
        print("METEOR SciGEN: {}".format(compute_meteor(dev_sci_ground, dev_sci_pred)))
        print("BERTSCORE QTSumm: {}".format(compute_bertscore(dev_qt_ground, dev_qt_pred)))
        print("BERTSCORE SciGEN: {}".format(compute_bertscore(dev_sci_ground, dev_sci_pred)))


        # print("-----Test-----")
        # test_qt_pred, test_qt_ground = eval_tapex(tokenizer, label_tokienizer, model, qt_test)
        # test_sci_pred, test_sci_ground = eval_tapex(tokenizer, label_tokienizer, model, sc_test)
        # print("BLEU QTSumm: {}".format(compute_bleu(test_qt_ground, test_qt_pred)))
        # print("BLEU SciGEN: {}".format(compute_bleu(test_sci_ground, test_sci_pred)))
        # print("ROUGE QTSumm: {}".format(compute_rouge(test_qt_ground, test_qt_pred)))
        # print("ROUGE SciGEN: {}".format(compute_rouge(test_sci_ground, test_sci_pred)))
        # print("METEOR QTSumm: {}".format(compute_meteor(test_qt_ground, test_qt_pred)))
        # print("METEOR SciGEN: {}".format(compute_meteor(test_sci_ground, test_sci_pred)))
        # print("BERTSCORE QTSumm: {}".format(compute_bertscore(test_qt_ground, test_qt_pred)))
        # print("BERTSCORE SciGEN: {}".format(compute_bertscore(test_sci_ground, test_sci_pred)))