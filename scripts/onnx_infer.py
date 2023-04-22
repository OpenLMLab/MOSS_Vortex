import torch
import onnx
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
import os
import onnxruntime as ort
import numpy as np
import time

class Inferer:
    def __init__(self, model_path, ):
        self.num_layers, self.heads, self.hidden, self.vocab_size = 34, 24, 256, 51200
        self.torchfloat, self.npfloat = torch.float16, np.float16

        self.ort_session = ort.InferenceSession(
            model_path,
            ort.SessionOptions(),
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        # self.ort_session.run()
        print('loaded')

    def top_k_top_p_filtering(self, logits, top_k, top_p, filter_value=-float("Inf"), min_tokens_to_keep=1, ):
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits


    def topk_search(self, 
                    input_ids, 
                    temperature=0.7, 
                    repetition_penalty = 1.1,
                    top_k = int(0),
                    top_p = 0.92,
                    max_time = 90,
                    max_iterations = 1024,
                    regulation_start = 512,
                    length_penalty = 1):
        self.bsz, self.seqlen = input_ids.shape
        raw_seqlen = input_ids.shape[1] 
        self.past_seqlen = 1
        input_ids, res = input_ids.cuda(), []
        presentkv = torch.zeros((self.num_layers * 2, self.bsz, self.heads, self.past_seqlen, self.hidden), dtype=self.torchfloat, device='cuda')

        queue_for_stop_word = []
        request_start_time = time.time()
        for i in range(int(max_iterations)):
            start_time = time.time()
            logits, presentkv = self.infer_(input_ids if i == 0 else new_generated_id, presentkv)
            print("infer cost", time.time() - start_time, "length", i+raw_seqlen)
            # print(logits.argmax(-1))
            logits = logits / temperature

            if repetition_penalty > 1:
                score = torch.gather(logits, 1, input_ids)
                # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
                score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                logits.scatter_(1, input_ids, score)

            filtered_logits = self.top_k_top_p_filtering(logits, top_k, top_p)
            probabilities = torch.softmax(filtered_logits, dim=-1)

            cur_len = input_ids.size(1)
            if cur_len > int(regulation_start):
                for i in [1279, 68, 12162, 29]:
                    probabilities[:, i] = probabilities[:, i] * pow(length_penalty, cur_len - regulation_start)
            # print(probabilities)
            new_generated_id = torch.multinomial(probabilities, 1)
            # print(new_generated_id)
            input_ids = torch.cat([input_ids, new_generated_id], dim=1)
            #res.append(new_generated_id.detach().cpu().long())
            # stop words components
            queue_for_stop_word.append(new_generated_id[0, 0].cpu().long())
            if len(queue_for_stop_word) > 3:
                queue_for_stop_word.pop(0)
            if queue_for_stop_word == [68, 12162, 29] or queue_for_stop_word == [68, 1219, 29]:
                break
            elif time.time() - request_start_time >= max_time-1:
                break
            elif i >= max_iterations:
                break

        return input_ids.cpu()


    def infer_(self, input_ids, pastkv):
        outputs_logits = torch.empty((self.bsz, self.vocab_size), dtype=torch.float32, device='cuda').contiguous()
        outputs_kvs = torch.empty(
            (self.num_layers * 2, self.bsz, self.heads, self.seqlen + self.past_seqlen, self.hidden),
            dtype=self.torchfloat,
            device='cuda'
        ).contiguous()
        attention_mask = torch.ones((self.bsz, self.seqlen + self.past_seqlen), dtype=torch.int64, device='cuda').contiguous()
        io_binding = self.ort_session.io_binding()

        assert input_ids.is_contiguous() and input_ids.dtype == torch.int64 and input_ids.size(1) == self.seqlen
        assert attention_mask.is_contiguous() and attention_mask.dtype == torch.int64 and attention_mask.size(1) == self.seqlen + self.past_seqlen
        for _ in range(self.num_layers * 2):
            assert pastkv[_].is_contiguous() and pastkv[_].dtype == self.torchfloat and pastkv[_].shape == torch.Size([self.bsz, self.heads, self.past_seqlen, self.hidden])
        for _ in range(self.num_layers * 2):
            assert outputs_kvs[_].is_contiguous() and outputs_kvs[_].dtype == self.torchfloat and outputs_kvs[_].shape == torch.Size([self.bsz, self.heads, self.seqlen + self.past_seqlen, self.hidden])
        assert outputs_logits.is_contiguous() and outputs_logits.dtype == torch.float32 and outputs_logits.shape == torch.Size([self.bsz, self.vocab_size])

        io_binding.bind_input(name='input_ids', device_type='cuda', device_id=0, element_type=np.int64,
                                          shape=input_ids.shape, buffer_ptr=input_ids.data_ptr())
        io_binding.bind_input(name='attention_mask', device_type='cuda', device_id=0, element_type=np.int64,
                              shape=attention_mask.shape, buffer_ptr=attention_mask.data_ptr())
        for _ in range(self.num_layers):
            io_binding.bind_input(name=f'past_key_values.{_}.key', device_type='cuda', device_id=0, element_type=self.npfloat,
                                  shape=pastkv[2 * _].shape, buffer_ptr=pastkv[2 * _].data_ptr())
            io_binding.bind_input(name=f'past_key_values.{_}.value', device_type='cuda', device_id=0, element_type=self.npfloat,
                                  shape=pastkv[2 * _ + 1].shape, buffer_ptr=pastkv[2 * _ + 1].data_ptr())

        io_binding.bind_output('logits', device_type='cuda', device_id=0, element_type=np.float32,
                                           shape=outputs_logits.shape, buffer_ptr=outputs_logits.data_ptr())

        for _ in range(self.num_layers):
            io_binding.bind_output(name=f'present.{_}.key', device_type='cuda', device_id=0, element_type=self.npfloat,
                                  shape=outputs_kvs[2 * _].shape, buffer_ptr=outputs_kvs[2 * _].data_ptr())
            io_binding.bind_output(name=f'present.{_}.value', device_type='cuda', device_id=0, element_type=self.npfloat,
                                  shape=outputs_kvs[2 * _ + 1].shape, buffer_ptr=outputs_kvs[2 * _ + 1].data_ptr())

        self.ort_session.run_with_iobinding(io_binding)
        # print(outputs_logits)
        self.past_seqlen += self.seqlen
        self.seqlen = 1

        return outputs_logits, outputs_kvs

