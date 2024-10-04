# multilingual-entity-insertion
Source code for "Entity Insertion in Multilingual Linked Corpora: The Case of Wikipedia"

## Code

There are two key parts to the code structure in this repo. The subdirectory `wikipedia_dumps_process` contains code related to the data processing. The subdirectory `data_modelling` contains code related to our data modeling framework.

### Data Processing Code

All the scripts needed to process the raw data are in the repo, in the subdirectory `wikipedia_dumps_process`. For the full processing pipeline, the scripts need to be chained in the correct order. The bash script `full_pipeline.sh` abstracts away the interactions between the scripts and can be used to run the entire data processing pipeline.

I leave the following recommendations when processing the data:
 - The script `test_data_version_finder.py` downloads the revision history before processing it. It can be configured to run with two different behaviors
   - Download all the missing revision history files: this will start the processing of the revision history from scratch
   - Only use the existing revision history files and don't download any additional ones: this is relevant to resume an interrupted job
 - No more than 3 processes can be used by `test_data_version_finder.py` to download the revision history, as otherwise we get blocked by Wikipedia. Additionally, sometimes the download processes hang. At this point, it is necessary to interrupt the script, delete the partial downloaded file, and resume the processing.
 - The crawler in `crawler/crawl_wiki.js` can download HTML pages from Wikipedia using multiple parallel threads. However, no more than 5 should be used. This is **very important** because Wikipedia will not prevent us from downloading the pages and no error will be thrown. However, all the HTML files downloaded will be "Server Error" HTML files, and not the content we want.
 - The script `test_data_generator.py` can be **very** memory-hungry especially for large languages (English, German, French, Spanish). I recommend using no more than 4 processes for the large language, while paying attention to the memory usage to avoid problems. If there is no rush, 2 processes are safe and can be left running unattended.
 - The script `input_generator_stage1.py` can also be memory-hungry. For the large languages, it is recommended to use the extra arguments provided in the script that significantly reduce the memory usage of the script.

### Modelling Code

The training infrastructure is present in `data_modelling/training`. Script `main.py` can be used to train a model for entity insertion, and it has a large number of command line arguments to allow for a versatile and easy training set-up. The training script expects a specific structure from the training data. Several examples of training datasets can be found in `/XXX/YYY/linkrec-llms/training/multilingual_datasets`. Keep in mind that the columns are not necessarily the same for stage 1 and stage 2.

The `training` directory contains multiple bash files that can be used to replicate the experiments in the paper.

#### Baselines

Our model is benchmarked against multiple baselines. We use
 - Random: Randomly rank the candidates.
 - String Match: Search for previously used mentions in the candidate text spans.
 - BM25: Apply BM25 on keywords extracted from the target lead paragraph and keywords extracted from the text span.
 - EntQA: Specifically, the retriever module from EntQA. The candidate text spans are encoded independently as well as the target entity. The text spans are ranked according to the cosine similarity between the embeddings of the text spans and the embeddings of the target article (see [paper](https://arxiv.org/abs/2110.02369)). Before EntQA can be used, the bash script `data_modelling/baselines/prepare_EntQA.sh` needs to be executed.
 - GET: As a generative model, we give each text candidate to the model and force it to generate the name for the target entity. The text candidates are then ranked by the score produced by the model for each generated text (see [paper](https://arxiv.org/abs/2209.06148)). Before GET can be used, the bash script `data_modelling/baselines/prepare_GET.sh` needs to be executed.
 - GPT-3.5/GPT-4: We give pairs of text contexts, where one element is the correct text span and the other element is an incorrect one. The model needs to predict which text span is most related to the target entity. **WARNING** this is an expensive baseline, use carefully.

The script `benchmark_baselines.py` is used to benchmark all baselines expect GPT, which is done using `benchmark_gpt.py`. The models are benchmarked using `benchmark_models.py`.

#### Benchmarking

Then there are several benchmarking subdirectories, relating to different benchmarking conditions.
 - `benchmark_real`: benchmarking done on the Simple English data. Particularly relevant for the ablations that were only conducted on Simple English.
 - `benchmark_multilingual`: benchmarking done on the full multilingual data. These are the most relevant (and most up-to-date) results.

 In order to generate the data for `benchmark_multilingual`, the script `data_modelling/benchmark_multilingual/merge_data.py` should be used. To generate the data for `benchmark_real`, the script `data_modelling/benchmark_multilingual/clean_data.py`.

## Data

The processed data is available [here](https://zenodo.org/records/13888211). Due to size constraints, we don't publish the full unprocessed dataset. If you are interested, please contact the authors.

## Models

All relevant models will be made available on HuggingFace.

## Relevant Notebooks
These are the most important notebooks in this repo.
 - `data_modelling/benchmark_multilingual/compare_models.ipynb`: Full results for the multilingual data (graphics and tables in LaTeX format)
 - `data_modelling/benchmark_multilingual/create_examples.ipynb`: Examples of predictions from baselines and models
 - `data_modelling/benchmark_real/compare_models_all.ipynb`: Results for Simple English (only ablations are relevant now)
 - `wikipedia_dumps_process/analysis_notebooks/added_links_analysis.ipynb`: Language statistics regarding the added links
 - `wikipedia_dumps_process/analysis_notebooks/graph_analysis.ipynb`: Language statistics regarding the number of pages and links available
