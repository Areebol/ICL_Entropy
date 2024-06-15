import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, TextStreamer, GenerationConfig
from transformers.models.opt.modeling_opt import OPTForCausalLM,OPTDecoderLayer
import os
import math
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

def entropy(probabilities):
    """
    return entropy of probabilitites
    """
    entropies = -torch.sum(probabilities * torch.log(probabilities + 1e-20), dim=-1)
    return torch.mean(entropies)

def attention_entropy(attentions_record: tuple[tuple[torch.Tensor]], layer: int, input_index: int) -> torch.Tensor:
    """
    attentions_record: word -> layer -> input -> attn_head -> raw_attention_maxtrix/vector
    """
    # 32*24*24 32*1*25 ... 32*1*199 </s>
    words_attention = [word_attention_record[layer][input_index] for word_attention_record in attentions_record]

    # 32*199*199
    tril_size = words_attention[-1].shape[-1]
    head_num = words_attention[0].shape[0]
    head_attention_matrix = torch.tril(torch.ones(head_num, tril_size, tril_size))

    count = 0
    for word_attention in words_attention:
        if word_attention.shape[1] == 1:
            head_attention_matrix[:, count, :(count+1)] = word_attention.squeeze(dim=1)
            count += 1
        else:
            for i in range(word_attention.shape[1]):
                head_attention_matrix[:, count, :(count+1)] = word_attention[:, i, :(count+1)]
                count += 1

    # 32 * 199
    log_len = torch.log(torch.arange(start=1, end=tril_size+1).repeat(head_num, 1)) + 1e-20
    head_entropy = torch.sum(- head_attention_matrix * torch.log(head_attention_matrix + 1e-20), dim=-1) / log_len
    return head_entropy

def get_attention_entropy(attn_matrix,soft_max=True,avg_head=False):
    with torch.no_grad():
        if soft_max==False:
            # Mask上三角
            for i in range(attn_matrix.shape[-1]):
                for j in range(attn_matrix.shape[-1]):
                    if j > i:  # j > i 表示在上三角部分
                        attn_matrix[:, i, j] = float("-inf")
            # 归一化
            attn_matrix = nn.functional.softmax(attn_matrix, dim=-1, dtype=torch.float32)
        if avg_head:
            logging.info("Avg_head on entropy use")
            attn_matrix= torch.mean(attn_matrix,dim=1).unsqueeze(1) # shape = (bs_size,#heads,len,len)
        else:
            logging.info("No Avg_head on entropy use")
            
        log_len = torch.log(torch.arange(start=1, end=attn_matrix.shape[-1]+1).repeat(attn_matrix.shape[0], 1)) + 1e-20
        head_entropy = torch.sum(- attn_matrix * torch.log(attn_matrix + 1e-20), dim=-1) / log_len
        return head_entropy

def response(tokenizer:AutoTokenizer, model:AutoModelForCausalLM, question:str, max_new_tokens:int=200):
    """
    return model's generation without decode
    """
    streamer = TextStreamer(tokenizer)
    gen_config = GenerationConfig(do_sample=False, num_beams=1)
    with torch.no_grad():
        inputs = tokenizer(question, padding=False, return_tensors='pt')
        input_ids = inputs['input_ids'].cuda()
        attention_mask = inputs['attention_mask'].cuda()
        res = model.generate(input_ids, attention_mask=attention_mask, generation_config=gen_config,
                             streamer=streamer, max_new_tokens=max_new_tokens)
    return res


def split_sentence(tokenizer,question,input_ids=None,split_words=['balls.','balls?','balls!','balls:','balls.\n','balls\n']):
    punctuation_id=(1 if 'Qwen' in tokenizer.name_or_path else 2)
    if input_ids is None:
        inputs = tokenizer(question, padding=False, return_tensors='pt')
        input_ids = inputs['input_ids'].squeeze().cpu()
    else:
        input_ids = input_ids.cpu()
    indices = []
    
    indices += [torch.nonzero(torch.eq(input_ids,tokenizer(split_word,padding=False,return_tensors='pt')['input_ids'].squeeze()[punctuation_id].cpu())).reshape(-1) for split_word in split_words]
    
    total_split = torch.cat(indices,dim=-1)
    sort_splits = torch.unique(torch.sort(total_split.view(-1)).values)
    logging.info(f"{split_words} {sort_splits}")
    return sort_splits

def crop_inputs(inputs,max_input_token,split_ids):
    input_ids = inputs['input_ids']
    # 无需裁剪
    if input_ids.shape[1] <= max_input_token:
        return inputs
    # 从裁剪处寻找句子结尾
    for i in range(max_input_token,0,-1):
        if i in split_ids:
            inputs['input_ids'] = inputs['input_ids'][:,:i]
            inputs['attention_mask'] = inputs['attention_mask'][:,:i]
            return inputs
    
    # 找不到完整句子，直接截断
    inputs['input_ids'] = inputs['input_ids'][:,:max_input_token]
    inputs['attention_mask'] = inputs['attention_mask'][:,:max_input_token]
    return inputs

def get_model_generate(tokenizer:AutoTokenizer, model:AutoModelForCausalLM, question:str, max_new_tokens:int=1, layer:int=-1, max_input_token=None, split_words=None):
    """返回对应层的每个token的entropy值"""
    gen_config = GenerationConfig(do_sample=False, num_beams=1,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id)
    with torch.no_grad():
        inputs = tokenizer(question, padding=False, return_tensors='pt')
        # 裁剪输入token数目在400以内 针对70b超大模型
        if max_input_token:
            if split_words:
                split_ids = split_sentence(tokenizer=tokenizer,question=question,split_words=split_words)
            else:
                split_ids = split_sentence(tokenizer=tokenizer,question=question)
                
            inputs = crop_inputs(inputs,max_input_token,split_ids)
        input_ids = inputs['input_ids'].cuda()
        assert input_ids.shape[0] == 1, "response_entropy暂时只能一个一个question的处理"
        
        attention_mask = inputs['attention_mask'].cuda()
        generate = model.generate(input_ids, attention_mask=attention_mask, generation_config=gen_config,
                             max_new_tokens=max_new_tokens, output_attentions=True, return_dict_in_generate=True,
                             output_hidden_states=True)
        generated_text = tokenizer.batch_decode(generate['sequences'], skip_special_tokens=True)
        prob_dim = generate['attentions'][-1][0].shape[-1]
        entropy_value = attention_entropy(generate['attentions'], layer, 0)
        res = {
            "generate": generate,
            "text": generated_text[0],
            "entropy": entropy_value,
            "prob_dim": prob_dim,
            "input_ids": input_ids,
            "attentions": generate['attentions'], # attention 矩阵
        }
        return res

def get_attention_k(model,layer_id:int=-1):
    """获取attention模块"""
    layers = model.model.layers
    return layers[layer_id].self_attn
    
def get_encoder_k(model,layer_id:int=-1):
    if isinstance(model,OPTForCausalLM):
        layers = model.model.decoder.layers
    elif 'QWenLMHeadModel' in model.__str__():
        layers = model.transformer.h
    else:
        layers = model.model.layers
    return layers[layer_id]
    
def get_hidden_state_k(res,k:int=-2):
    """获取倒数第k层的embedding"""
    hidden_states = res['hidden_states'][0]
    hidden_state_k = hidden_states[k]
    return hidden_state_k

def get_attention_matrix(self,hidden_states,soft_max=True):
    """
    self: llama encoder模块
    hidden_states: embedding input
    """
    self.eval()
    bsz, q_len, _ = hidden_states.size()
    position_ids = torch.arange(0,q_len).reshape(1,-1)
    # 创建全零矩阵
    attention_mask = torch.zeros(bsz, 1, q_len, q_len)
    # 生成下三角掩码
    for i in range(q_len):
        for j in range(q_len):
            if j > i:  # j > i 表示在上三角部分
                attention_mask[:, :, i, j] = float('-inf')
    if soft_max == False:
        logging.warn("Not soft_max on attention weight is using")
    else:
        logging.warn("Soft_max on attention weight is using")
    if isinstance(self,OPTDecoderLayer):
        layer_outputs = self(
        hidden_states,
        attention_mask=attention_mask,
        output_attentions=True,
        )
    elif "QWenBlock" in self.__str__():
        # dtype
        dtype = next(self.parameters()).dtype
        hidden_states = hidden_states.to(dtype)
        attention_mask = attention_mask.to(dtype)
        layer_outputs = self(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=True,
            soft_max=soft_max, # !!! 不做归一化
            ) 
    else:
        # half
        dtype = next(self.parameters()).dtype
        hidden_states = hidden_states.to(dtype)
        attention_mask = attention_mask.to(dtype)
        layer_outputs = self(
        hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        output_attentions=True,
        soft_max=soft_max, # !!! 不做归一化
        )
    # return self_attn_weights
    return layer_outputs[1]

  
def get_sentences_token_by_max(values,splits):
    max_indexs = []
    start_index = torch.tensor(0)
    for end_index in splits:
        sub_tensor = values[start_index:end_index]

        # 找到子 tensor 内的最大值的下标
        max_index = torch.argmax(sub_tensor)

        # 计算相对于整个 values tensor 的绝对下标
        absolute_index = start_index.item() + max_index.item()
        max_indexs.append(absolute_index)
        start_index = end_index
    return max_indexs

def get_attention_matrix_k_to_last(self,res,k=1,soft_max=True):
    """计算倒数第k层的attention矩阵"""
    # hidden_states = res['hidden_states'][0]
    # hidden_states = hidden_states[-(k+1)]
    hidden_states = get_hidden_state_k(res,-(k+1))
    encoder_k_to_last = get_encoder_k(self,-k)
    return get_attention_matrix(encoder_k_to_last,hidden_states,soft_max=soft_max)

def split_attention_matrix(attn_matrix,split_ids):
    """
    切分attention_matrix
    返回切分后的sub attn_matrix
    返回各个token绝对下标
    """
    split_attn_matrixs = []
    split_tokens_indexs = []
    end_ids = split_ids
    if len(split_ids) == 0:
        split_attn_matrixs.append(attn_matrix)
        split_tokens_indexs.append(np.arange(0,attn_matrix.shape[-1]))  
        return split_attn_matrixs,split_tokens_indexs
    if isinstance(split_ids,torch.Tensor):
        end_ids = end_ids.tolist()
    start_id = 0
    if end_ids[-1] < attn_matrix.shape[-1] - 1:
        # 保证切分所有句子
        end_ids.append(attn_matrix.shape[-1]-1)
        
    for end_id in end_ids:
        sub_matrix = attn_matrix[:,:,start_id:end_id+1,start_id:end_id+1]
        sub_token_ids = np.arange(start_id,end_id+1)
        split_attn_matrixs.append(sub_matrix)
        split_tokens_indexs.append(sub_token_ids)
        start_id = end_id+1
    return split_attn_matrixs,split_tokens_indexs

def get_token_weight(attn_matrix):
    num_head,q_len = attn_matrix.shape[-3],attn_matrix.shape[-1]
    if q_len<=1:
        return torch.tensor(1)
    # 每个头
    # mean_attn_matrix = torch.mean(attn_matrix,dim=1).squeeze()
    total_token_weight = torch.zeros(q_len)
    for head_id in range(attn_matrix.shape[1]):
        token_weight = []
        mean_attn_matrix = attn_matrix[:,head_id,:,:].squeeze()
        for i in range(q_len):
            token_weight.append(torch.sum(mean_attn_matrix[:,i])/(q_len-i))
        token_weight = torch.tensor(token_weight)
        # 归一化
        token_weight = nn.functional.softmax(token_weight, dim=-1, dtype=torch.float32)
        total_token_weight += token_weight
    return total_token_weight / q_len

def get_token_weights(attn_matrixs):
    """计算矩阵的权重"""
    token_weights = []
    for attn_matrix in attn_matrixs:
        token_weights.append(get_token_weight(attn_matrix))
    return token_weights

def weighted_hidden_states(weights,token_ids,res):
    hidden_state = get_hidden_state_k(res,-2)
    weighted_hidden_states = []
    for weight,token_id in zip(weights,token_ids):
        weighted_states = weight.view(1,-1,1).to(hidden_state.device) * hidden_state[:,token_id,:]
        weighted_hidden_states.append(torch.sum(weighted_states,dim=1))
    return torch.stack(weighted_hidden_states,dim=1)

def split_attn_matrix(model,res,sort_splits,soft_max=True):
    # 计算最后一层attention矩阵
    attn_matrix = get_attention_matrix_k_to_last(model,res,k=1,soft_max=soft_max)
    # 切割attention矩阵为sentence子矩阵
    sentence_attn_matrixs,token_ids = split_attention_matrix(attn_matrix,sort_splits)
    # 计算各个attn_matrix的权重
    weights = get_token_weights(sentence_attn_matrixs)
    return weights,token_ids