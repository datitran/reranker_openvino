from pathlib import Path
from transformers import AutoTokenizer
from optimum.onnxruntime import (
    ORTOptimizer,
    ORTModelForSequenceClassification,
    ORTQuantizer,
)
from optimum.onnxruntime.configuration import OptimizationConfig, AutoQuantizationConfig

# Define model path and load the model
model_id = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
onnx_path = Path("cross-encoder-onnx-model")

# Convert the model to ONNX
model = ORTModelForSequenceClassification.from_pretrained(
    model_id, from_transformers=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    use_fast=True,
)

# Save the ONNX model
model.save_pretrained(onnx_path)
tokenizer.save_pretrained(onnx_path)

# Optimize the model
optimizer = ORTOptimizer.from_pretrained(model)
optimization_config = OptimizationConfig(optimization_level=99)

optimizer.optimize(
    save_dir=onnx_path,
    optimization_config=optimization_config,
)

# Quantize the optimized model
dynamic_quantizer = ORTQuantizer.from_pretrained(model)
dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

# Apply quantization
model_quantized_path = dynamic_quantizer.quantize(
    save_dir=onnx_path,
    quantization_config=dqconfig,
)
