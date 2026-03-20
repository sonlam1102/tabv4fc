# TabV4FC - Boosting LLMs for Tabular Fact-checking via Table Verbalization 

## Datasets: 
- Table-to-text generation: QTSUMM and SCIGEN
- Tabular Fact-checking: TabFACT, SCITAB and PubHealthTAB

## Data availability:
- QTSUMM: https://huggingface.co/datasets/yale-nlp/QTSumm
- SCIGEN: https://huggingface.co/datasets/kasnerz/scigen
- TabFACT: https://github.com/wenhuchen/Table-Fact-Checking
- SCITAB: https://github.com/XinyuanLu00/SciTab
- PubHealthTAB: https://github.com/mubasharaak/PubHealthTab
  
## Source codes: 
- To fine-tune TAPEX for Table-to-text generation: run the bash script fine_tune_tapex.sh
- To create the generation text that describes the table: run the bash script infer_tapex.sh
- To perform Tabular Fact-checking by LLMs: run the bash script run_llms.sh
We provided the sample summarized data in results/summary/ directory. You can used it to run LLMs for tabular Fact-checking.

## Contributors:
Son Thanh Luu - Japan Advanced Institute of Science and Technology (JAIST)   
Trung Vo - Japan Advanced Institute of Science and Technology (JAIST)   
Vu Tran - Japan Advanced Institute of Science and Technology (JAIST) - Advisor     
Prof. Tomoko Matsui - The Institute of Statistical Mathematics (ISM) - Supervisor    
Prof. Minh Le Nguyen - Japan Advanced Institute of Science and Technology (JAIST) - Supervisor    

## Publication: 
https://link.springer.com/article/10.1007/s41060-025-00998-3

```
@article{Luu2026BoostingLLMFactChecking,
  author    = {Luu, S. T. and Vo, T. and Tran, V. and others},
  title     = {Boosting large-language models for fact-checking: leveraging verbalized tabular data as evidence},
  journal   = {International Journal of Data Science and Analytics},
  volume    = {22},
  pages     = {109},
  year      = {2026},
  doi       = {10.1007/s41060-025-00998-3}
}
``` 

