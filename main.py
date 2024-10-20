import argparse
import numpy as np
import pandas as pd
from typing import Any
from openvino.runtime import Core
from transformers import AutoTokenizer


def load_openvino_model(model_path: str) -> Core:
    core = Core()
    model = core.read_model(model_path + ".xml")
    compiled_model = core.compile_model(model, "CPU")
    return compiled_model


def rerank(
    compiled_model: Core,
    query: str,
    results: list[str],
    tokenizer: AutoTokenizer,
    batch_size: int,
) -> np.ndarray[np.float32, Any]:
    max_length = 512
    all_logits = []

    # Split results into batches
    for i in range(0, len(results), batch_size):
        batch_results = results[i : i + batch_size]
        inputs = tokenizer(
            [(query, item) for item in batch_results],
            padding=True,
            truncation="longest_first",
            max_length=max_length,
            return_tensors="np",
        )

        # Extract input tensors (convert to NumPy arrays)
        input_ids = inputs["input_ids"].astype(np.int32)
        attention_mask = inputs["attention_mask"].astype(np.int32)
        token_type_ids = inputs.get("token_type_ids", np.zeros_like(input_ids)).astype(
            np.int32
        )

        infer_request = compiled_model.create_infer_request()
        output = infer_request.infer(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )

        logits = output["logits"]
        all_logits.append(logits)

    all_logits = np.concatenate(all_logits, axis=0)
    return all_logits


def fetch_search_data(search_text: str) -> pd.DataFrame:
    # Usually you would fetch the data from a database
    df = pd.read_csv("cnbc_headlines.csv")
    df = df[~df["Headlines"].isnull()]

    texts = df["Headlines"].tolist()

    # Load the model and rerank
    openvino_model = load_openvino_model("cross-encoder-openvino-model/model")
    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-TinyBERT-L-2-v2")
    rerank_scores = rerank(openvino_model, search_text, texts, tokenizer, batch_size=16)

    # Add the rerank scores to the DataFrame and sort by the new scores
    df["rerank_score"] = rerank_scores
    df = df.sort_values(by="rerank_score", ascending=False)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch search results with reranking using OpenVINO"
    )

    parser.add_argument(
        "--search_text",
        type=str,
        required=True,
        help="The search text to use for reranking",
    )

    args = parser.parse_args()

    df = fetch_search_data(args.search_text)
    print(df)
