```python
from openai import OpenAI
from turboquant_rag import TurboQuantCompressor

client = OpenAI()

def embed(text: str) -> list[float]:
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return res.data[0].embedding


quantizer = TurboQuantCompressor.new(
    dim=1536,
    angle_bits=4,
    projections=32,
    seed=42,
)

# ingest
doc_vec = embed("some chunk text")
doc_code = quantizer.encode(doc_vec)

# query
query_vec = embed("user query")
score = quantizer.inner_product_estimate(doc_code, query_vec)
```