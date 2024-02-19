from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

def rank_contexts(contexts, target_title, model, tokenizer):
    force_words = [target_title]
    force_words_ids = tokenizer(force_words)
    
    scores = []
    with torch.no_grad():
        for context in contexts:
            input_ids = tokenizer(context, return_tensors='pt', truncation=True).to('cuda')
            gen_tokens = model.generate(**input_ids,
                                        max_length=len(force_words_ids['input_ids'][0]),
                                        num_beams=2,
                                        early_stopping=True,
                                        min_length=len(force_words_ids['input_ids'][0]),
                                        output_scores=True,
                                        return_dict_in_generate=True,
                                        force_words_ids=[force_words_ids['input_ids'][0][:-1]]
            )
            scores.append(gen_tokens.sequences_scores[0].item())
    return scores