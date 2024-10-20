# Rerank Search Results with OpenVINO

## Getting Started

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the reranker:

```
python main.py --search_text "best stock"
```

## Notes

* The reranker model `cross-encoder/ms-marco-TinyBERT-L-2-v2` has already been optimized and converted to OpenVINO. You can take any model from HuggingFace and then use the `save_model.py` to save it to `ONNX`. Once you have it you can do this:

    ```
    mo --input_model cross-encoder-onnx-model/model.onnx --output_dir cross-encoder-openvino-model
    ```

    This will then convert the model to OpenVINO

## LICENSE

See [LICENSE](LICENSE) for details.
Copyright (c) 2024 [Dat Tran](http://www.dat-tran.com/).
