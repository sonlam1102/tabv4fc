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

## Publication: 
tba 

