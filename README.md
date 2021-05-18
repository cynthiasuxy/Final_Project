CRF for Target Extraction of COVID-19 Events on Twitter
---------------------------------------------------
_____________
*ANLY521 Final Project by Xinyi Su*

This project uses a CRF model to extract relevant COVID-19 events from Twitter. The data is collected from Twitter using 
Twitter API and the script from W-NUT 2020 Shared Task #3 , for the two COVID-19 events “test positive” and “death”. The 
goal of this study is to use CRF model to figure out whether the keywords for the relevant COVID-19 event indicate a real 
COVID case in life or a general comment on the COVID-19 event. Therefore, to simplify the task, I tag the keywords for 
each event and convert Target Extraction into a classification task. The features added to the model include word type, 
word POS, word shape, and the word itself. 


Data is extracted from the W-NUT 2020 Shared Task (http://noisy-text.github.io/2020/extract_covid19_event-shared_task.html).
Please contact yzzongshi@gmail.com and cc alan.ritter@cc.gatech.edu (the authors of the task) to
get access to the full annotated dataset.  


## Script
The script is used to generate the CRF results in this paper: [url]  

Please edit the configuration as:  
`python crf.py --test_positive_path data/positive-add_text.jsonl --death_case_path data/death-add_text.jsonl`

## Results

The accuracy score for the whole corpus including both events using the CRF models are 0.71 and 0.77 for the base 
feature set and the other feature set including those of previous and subsequent words respectively.
For each specific event, please find the scores listed in the table below.

|Event  | Model Name | Accuracy | Precision | Recall|F1|
| ---------- | -------- | --------- | ------- | ---|---|
| Test Positive | CRF (base)| 0.73|0.755|0.916|0.828|
| Test Positive | CRF (add neighbor features)| 0.656|0.775|0.294|0.426|
| Death | CRF (base)| 0.777 |0.802|0.909|0.852|
| Death | CRF (add neighbor features)| 0.746|0.788|0.553|0.65|
  
