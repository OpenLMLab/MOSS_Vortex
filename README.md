# MOSS Vortex

Moss Vortex is a lightweight, fast, simple, and high-performance deployment and inference backend engineered specifically for MOSS 003, providing a wealth of features aimed at enhancing performance and functionality, built upon the foundations of MOSEC and Torch.  

You only need to execute a few commands and spend a few minutes to quickly deploy your MOSS 003 inference server on your own GPU server.


The features encompass:

* Websocket-based streaming output: MOSS Vortex utilizes Websockets to enable real-time, bidirectional communication between the server and clients. This allows for efficient streaming of output, providing faster response times and improved user experience.

* Multiple sampling strategies for LLM generation: The application supports various sampling strategies for Large Language Models (LLMs) to improve the quality and diversity of generated content. This allows for better control over the output and helps fine-tune the results based on specific requirements.

* Infinite conversation loops: MOSS Vortex is designed to handle extended dialogues between users and the AI, facilitating engaging and dynamic conversations without any limitations on the number of exchanges.

* Support for custom tools: The application offers support for multiple custom tools, allowing users to integrate and utilize additional functionality based on their needs. This flexibility ensures that MOSS Vortex can adapt to a wide range of use cases and requirements.

* ONNX model acceleration: MOSS Vortex takes advantage of ONNX (Open Neural Network Exchange) format for model acceleration, optimizing the performance of the underlying deep learning models. This ensures faster inference times and more efficient resource utilization.

* Model parallelism: The application leverages model parallelism techniques to distribute the workload across multiple GPUs or other processing units. This allows for improved scalability and performance, particularly when dealing with large-scale models and data.  

The main flaw of MOSS Vortex is does not implement _Token Batching_, which is crucial for LLM reasoning, and I will implement it shortly.  


<img src="./img/search_case.jpeg" alt="example" width="400" height="400">


## QuickStart

To quickly deploy Moss Vortex using Docker:  


```
git clone https://github.com/piglaker/vortex.git  
cd Vortex  
bash install_run.sh     
```
## Test
To run a test on MOSS Vortex:  
```
bash scripts/test.sh
```  

## Interface

```
curl -X POST http://127.0.0.1:21333/inference -d \
'{"x": "<|Human|>: hello<eoh>\n<|Inner thoughts|>: None.<eot>\n<|Commands|>: None<eoc>\n<|Results|>: None<eor>\n<|MOSS|>:", \  
"max_iterations":"128", \  
"temperature":"0.7", \
"repetition_penalty":"1.1"\
}'
```

Return Format: 
```
>> bash scripts/short_vortex_test.sh
#date
{
  "pred": "<|Human|>: hello<eoh>\n<|Inner Thoughts|>: None.<eot>\n<|Commands|>: None<eoc>\n<|Results|>: None<eor>\n<|MOSS|>: Hello! How may I assist you today?<eom>", \
"input_token_num": 331, \
"new_generations_token_num": 10 \
"new_generations": " Hello! How may I assist you today?<eom>"
}
#date
```


## Metrics
To check the metrics:  
```
http 127.0.0.1:21333/metrics
```

Logs Format:
```
2023-04-18 00:50:46,707 - 210 - INFO - mosec_server.py:652 - <|Human|>: 写一段python快排代码<eoh>
<|Inner Thoughts|>: None<eot>
<|Commands|>: None<eoc>
<|Results|>: None<eor>
<|MOSS|>: 这里是一个简单的Python快速排序的代码示例：

`python
def quick_sort(arr):
    if len(arr) <= 1:
        return arr

    pivot = arr[0]  # 选择第一个元素作为基准点
    left = [x for x in arr[1:] if x <= pivot]
    right = [x for x in arr[1:] if x > pivot]

    return quick_sort(left) + right

# 示例
print(quick_sort([3, 6, 8, 10, 1, 2]))
`

输出结果为 `[1, 2, 3, 6, 8, 10]`。<eom>
2023-04-18 00:51:30,113 - 213 - INFO - mosec_server.py:743 - [MOSEC] [FORWARD] First Token Generation Cost: 0.09637761116027832
2023-04-18 00:51:30,461 - 213 - INFO - mosec_server.py:747 - [MOSEC] [FORWARD] Recent Token Generation Cost: 0.04331459999084473
2023-04-18 00:51:30,912 - 213 - INFO - mosec_server.py:542 - [MOSEC] [STREAM] Graceful close websockets 
2023-04-18 00:51:30,912 - 213 - INFO - mosec_server.py:623 - [MOSEC] [INFER] Request Cost: 0.8990638256072998
```

## Configuration

The following configurations are available for Moss Vortex:  
- CUDA Version: 11.7  
- GPU: 8 * A800 (Recommended)  
- Default Batch Size: 8   
- Default Wait Time for Batching: 10  
- Default Infer Timeout: 70,000 ms  
- Port: 21333 (used for Nginx)  
- Mosec Version: 0.6  


## Citation
If you use Moss Vortex in your work, please cite it as follows:
```
@software{MOSS_Vortex2023,  
  title = {{Moss Vortex: An advanced deployment and inference backend for MOSS based on MOSEC and Torch}},  
  author = {Xiaotian Zhang, Zhengfu He, Tianxiang Sun},  
  url = {https://github.com/piglaker/Vortex},  
  year = {2023}  
}
```



