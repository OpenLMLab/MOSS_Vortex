date
curl -X POST http://127.0.0.1:7003/inference -d '{"x": "<|Human|>: hello<eoh>\n<|Inner Thoughts|>: None.<eot>\n<|Commands|>: None<eoc>\n<|Results|>: None<eor>\n<|MOSS|>:", "max_iterations":"128", "temperature":"0.7", "repetition_penalty":"1.1"}'
date