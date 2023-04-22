
date
curl -X POST http://127.0.0.1:7003/inference -d '{"x": "<|Human|>: hello? Could you teach me how to programm quick sort in python ? <eoh><|Inner Thoughts|>: This question requires external knowledge about Python programming Quick Sort algorithm<eot><|Commands|>: None<eoc><|Result|>: None<eor>", "max_iterations":"512"}'
date