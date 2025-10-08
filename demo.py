from read_data import *
import torch
from transformers import TapexTokenizer, BartForConditionalGeneration, BartTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from llms import *


if __name__ == '__main__':
    train = read_tabfact('/home/sonlt/drive/data/tabfact', type="test")
    print(len(train))
    
    sum_model = TableFactSum(model="tapex_large_200")

    sample_data = train[0:3]
    sum_model.make_features(train[0:3])
    output = sum_model.generate_summary()
    
    for i in range(0, len(sample_data)):
        print("------------")
        print("Claim: {}\n".format(sample_data[i][0]))
        print("Table: \n")
        print(sample_data[i][1])
        sample_data[i][1].to_csv("sample_tabfact_{}.csv".format(i))
        print("Table summarization: {}\n".format(output[i]))
        print("Label: {}".format(sample_data[i][2]))

        print("------------")

    
    # train_scitab = read_scitab('/home/sonlt/drive/data/scitab')
    # sum_model2 = SciTabSum(model="tapex_large_300")
    # sample_data2 = train_scitab[0:3]
    # sum_model2.make_features(train_scitab[0:3])
    # output2 = sum_model2.generate_summary()
    # for i in range(0, len(sample_data2)):
    #     print("------------")
    #     print("Claim: {}\n".format(sample_data2[i]["claim"]))
    #     print("Table: \n")
    #     print(sample_data2[i]["table"])
    #     sample_data2[i]["table"].to_csv("sample_scitab_{}.csv".format(i))
    #     print("Table summarization: {}\n".format(output2[i]))
    #     print("Label: {}".format(sample_data2[i]["label"]))

    #     print("------------")
