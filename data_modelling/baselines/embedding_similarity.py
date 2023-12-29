from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm

def rank_contexts(model_name, contexts, target_title, target_lead, mentions = None):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    if mentions is None:
        target_input = f'{target_title} {target_lead}'
    else:
        target_input = f'{target_title} {" ".join(mentions)} {target_lead}'
    target_input = tokenizer([target_input], return_tensors='pt', padding=True, truncation=True)
    target_input = target_input.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        target_output = model(**target_input)[0][:, 0].squeeze()
        context_outputs = []
        for i in range(0, len(contexts), 36):
            context_inputs = contexts[i:i+36]
            context_inputs = tokenizer(context_inputs, return_tensors='pt', padding=True, truncation=True)
            context_inputs = context_inputs.to('cuda' if torch.cuda.is_available() else 'cpu')            
            context_output = model(**context_inputs)[0][:, 0]
            for j in range(len(context_output)):
                context_outputs.append(context_output[j])
    
    scores = torch.nn.functional.cosine_similarity(target_output, torch.stack(context_outputs)).tolist()
    return scores
    