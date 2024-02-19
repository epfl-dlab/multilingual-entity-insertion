from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
import torch.nn as nn
from collections import OrderedDict
from transformers import BertModel
import json

class DualEncoder(nn.Module):
    def __init__(self, mention_encoder,
                 entity_encoder,
                 type_loss):
        super(DualEncoder, self).__init__()
        self.mention_encoder = mention_encoder
        self.entity_encoder = entity_encoder

    def encode(self, mention_token_ids=None,
               mention_masks=None,
               candidate_token_ids=None,
               candidate_masks=None,
               entity_token_ids=None,
               entity_masks=None):
        candidates_embeds = None
        mention_embeds = None
        entity_embeds = None
        # candidate_token_ids and mention_token_ids not None during training
        # mention_token_ids not None for embedding mentions during inference
        # entity_token_ids not None for embedding entities during inference
        if candidate_token_ids is not None:
            B, C, L = candidate_token_ids.size()
            candidate_token_ids = candidate_token_ids.view(-1, L)
            candidate_masks = candidate_masks.view(-1, L)
            # B X C X L --> BC X L
            candidates_embeds = self.entity_encoder(
                input_ids=candidate_token_ids,
                attention_mask=candidate_masks
            )[0][:, 0, :].view(B, C, -1)
        if mention_token_ids is not None:
            mention_embeds = self.mention_encoder(
                input_ids=mention_token_ids,
                attention_mask=mention_masks
            )[0][:, 0, :]
        if entity_token_ids is not None:
            # for getting all the entity embeddings
            entity_embeds = self.entity_encoder(input_ids=entity_token_ids,
                                                attention_mask=entity_masks)[
                                0][:, 0, :]
        return mention_embeds, candidates_embeds, entity_embeds

    def forward(self,
                mention_token_ids=None,
                mention_masks=None,
                candidate_token_ids=None,
                candidate_masks=None,
                entity_token_ids=None,
                entity_masks=None
                ):
        """

        :param inputs: [
                        mention_token_ids,mention_masks,  size: B X L
                        candidate_token_ids,candidate_masks, size: B X C X L
                        passages_labels, size: B X C
                        ]
        :return: loss, logits

        """
        return self.encode(mention_token_ids, mention_masks,
                            candidate_token_ids, candidate_masks,
                            entity_token_ids, entity_masks)
        
def load_model(is_init, config_path, model_path, device, type_loss,
               blink=True):
    with open(config_path) as json_file:
        params = json.load(json_file)
    if blink:
        ctxt_bert = BertModel.from_pretrained(params["bert_model"])
        cand_bert = BertModel.from_pretrained(params["bert_model"])
    else:
        ctxt_bert = BertModel.from_pretrained('bert-large-uncased')
        cand_bert = BertModel.from_pretrained('bert-large-uncased')
    state_dict = torch.load(model_path) if device.type == 'cuda' else \
        torch.load(model_path, map_location=torch.device('cpu'))
    if is_init:
        if blink:
            ctxt_dict = OrderedDict()
            cand_dict = OrderedDict()
            for k, v in state_dict.items():
                if k[:26] == 'context_encoder.bert_model':
                    new_k = k[27:]
                    ctxt_dict[new_k] = v
                if k[:23] == 'cand_encoder.bert_model':
                    new_k = k[24:]
                    cand_dict[new_k] = v
            ctxt_bert.load_state_dict(ctxt_dict, strict=False)
            cand_bert.load_state_dict(cand_dict, strict=False)
        model = DualEncoder(ctxt_bert, cand_bert, type_loss)
    else:
        model = DualEncoder(ctxt_bert, cand_bert, type_loss)
        model.load_state_dict(state_dict['sd'])
    return model

def rank_contexts(contexts, target_title, target_lead, model, tokenizer):
    target_input = f'{target_title} {target_lead}'
    target_input = tokenizer([target_input], return_tensors='pt', padding=True, truncation=True).to('cuda')
    
    with torch.no_grad():
        _, _, target_embedding = model.forward(entity_token_ids=target_input['input_ids'],
                                               entity_masks=target_input['attention_mask'])
        target_embedding = target_embedding.squeeze().detach().cpu()
        context_embeddings = []
        for context in contexts:
            context_input = tokenizer([context], return_tensors='pt', padding=True, truncation=True).to('cuda')
            _ , _, context_embedding = model.forward(entity_token_ids=context_input['input_ids'],
                                                    entity_masks=context_input['attention_mask'])
            context_embedding = context_embedding.squeeze().detach().cpu()
            context_embeddings.append(context_embedding)    
    scores = torch.nn.functional.cosine_similarity(target_embedding, torch.stack(context_embeddings)).tolist()
    return scores