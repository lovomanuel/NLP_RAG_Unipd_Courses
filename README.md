# Retrieval Augmented Generation on UniPD's courses information.
This is a project developed as a part of the course in NLP from University of Padua.
The aim was to develop a RAG system on a dataset of our choice.
## Dataset
We have chosen to create our own dataset since we believe that it could be useful to have an assistant on information regarding courses at University of Padua.
It's always difficult to find information relative to mandatory exams, study plan, pre-requisites, so we decided to create our own dataset, using the notebook "NicolaLorenzon_ManuelLovo_NLP_DATASET_GENERATOR.ipynb".
## RAG
After an important pre-processing of the dataset, we have developed a quantitative method for doing model-selection in order to select the best combination of vector-db, embedder, type of dataset, chunk-size etc.
We have generated, by using SoTA model GPT-4o, hundreds of questions from our dataset and we have seen for which combination of paramaters the model retrieves most information.
Then, we have tested with different LLM, prompt and temperature and we have qualitatively decided which was the best one.
