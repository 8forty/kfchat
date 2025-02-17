#  print(onnxruntime.get_available_providers())
# results on medusa:
# ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
import timeit

from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2
from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import SentenceTransformerEmbeddingFunction


def onnx_minilm(docs):
    ONNXMiniLM_L6_V2(preferred_providers=['CUDAExecutionProvider'])(docs)


#         device (str, optional): Device (like "cuda", "cpu", "mps", "npu") that should be used for computation. If None, checks if a GPU can be used.
# def st_cuda_minilm(docs):
#     SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2', device='cuda')(docs)


def st_cpu_minilm(docs):
    SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2', device='cpu')(docs)


def st_mps_minilm(docs):
    SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2', device='mps')(docs)


def st_gpu_minilm(docs):
    SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2', device='gpu')(docs)


def st_npu_minilm(docs):
    SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2', device='npu')(docs)


# def st_cuda_mpnet(docs):
#     SentenceTransformerEmbeddingFunction(model_name='all-mpnet-base-v2', device='cuda')(docs)


def st_cpu_mpnet(docs):
    SentenceTransformerEmbeddingFunction(model_name='all-mpnet-base-v2', device='cpu')(docs)


def st_mps_mpnet(docs):
    SentenceTransformerEmbeddingFunction(model_name='all-mpnet-base-v2', device='mps')(docs)


def st_gpu_mpnet(docs):
    SentenceTransformerEmbeddingFunction(model_name='all-mpnet-base-v2', device='gpu')(docs)


def st_npu_mpnet(docs):
    SentenceTransformerEmbeddingFunction(model_name='all-mpnet-base-v2', device='npu')(docs)


if __name__ == "__main__":
    mdocs = []
    for i in range(1000):
        mdocs.append(f"this is a document with id {i}")

    for f in [onnx_minilm, st_cpu_minilm, st_mps_minilm, st_gpu_minilm, st_npu_minilm, st_cpu_mpnet, st_mps_mpnet, st_gpu_mpnet, st_npu_mpnet]:
        start = timeit.default_timer()
        f(mdocs)
        print(f"{f.__name__} time: {timeit.default_timer() - start:.2f}s")
