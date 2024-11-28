# **Building Production-Grade RAG Applications: Addressing Common Pain Points**

## **Introduction**

- **Overview**: Prototyping a Retrieval-Augmented Generation (RAG) application is straightforward, but building one that is performant, robust, and scalable for a large knowledge corpus is challenging.

- **Objective**: This presentation addresses common pain points when building production-grade RAG applications and provides detailed solutions, including advanced optimization techniques.

---

## **Pain Point 1 - Missing Content**

### **Problem**

- **Description**: The RAG system provides plausible but incorrect answers when the real answer is not present in the knowledge base. This leads to users receiving misleading information.

### **Solutions**

#### **1. Data Cleaning**

- **Explanation**: Ensure the data source is accurate and free of conflicting information. Poor data quality leads to incorrect results.

- **Steps to Follow**:
  - **Data Verification**: Regularly audit and update your knowledge base.
  - **Data Cleaning**: Remove duplicates, correct errors, and resolve inconsistencies.

#### **2. Prompt Enhancement**

- **Explanation**: Create prompts that encourage the model to acknowledge when it doesn't know an answer.

- **Steps to Follow**:
  - **Explicit Instructions**: Use prompts like "If you're not sure of the answer, please respond with 'I don't know'."
  - **Examples**: Provide examples of prompts and expected responses.

#### **3. Pydantic Programs**

- **Explanation**: Use Pydantic programs to define structured output schemas.

### **Example: Pydantic Programs Implementation**

```python
from pydantic import BaseModel
from llama_index.program import OpenAIPydanticProgram

class Song(BaseModel):
    title: str
    duration_seconds: int

class Album(BaseModel):
    name: str
    artist: str
    songs: List[Song]

program = OpenAIPydanticProgram.from_defaults(
    output_cls=Album,
    prompt_template_str="Generate a sample album with songs inspired by {movie_name}.",
)
output = program(movie_name="Inception")
```

---

## **Pain Point 2 - Key Documents Omitted**

### **Problem**

- **Description**: Essential documents don't appear in the top results returned by the retrieval component.

### **Solutions**

#### **1. Hyperparameter Tuning**

- **Explanation**: Adjust parameters like `chunk_size` and `similarity_top_k` to improve retrieval effectiveness.

- **Steps to Follow**:
  - **Use `ParamTuner`**: Use LlamaIndex's `ParamTuner` to automate tuning.
  - **Experimentation**: Test different values to find optimal settings.

### **Code Example: Parameter Tuning**

```python
param_dict = {"chunk_size": [256, 512, 1024], "top_k": [1, 2, 5]}
param_tuner = ParamTuner(
    param_fn=objective_function_semantic_similarity,
    param_dict=param_dict,
    fixed_param_dict=fixed_param_dict,
    show_progress=True,
)
results = param_tuner.tune()
```

#### **2. Reranking**

- **Explanation**: Implement reranking strategies to reorganize retrieval results before sending them to the language model.

- **Steps to Follow**:
  - **Increase similarity_top_k**: Retrieve more documents to provide a broader base for reranking.
  - **Use Rerankers**: Integrate tools like CohereRerank.

### **Example: CohereRerank Implementation**

```python
from llama_index.postprocessor.cohere_rerank import CohereRerank

cohere_rerank = CohereRerank(api_key='YOUR_COHERE_API_KEY', top_n=2)
query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[cohere_rerank],
)
response = query_engine.query("Your question here")
```

---

## **Pain Point 3 - Not in Context**

### **Problem**

- **Description**: Retrieved documents containing the answer don't make it into the context used to generate the response, due to consolidation process limitations.

### **Solutions**

#### **1. Adjust Retrieval Strategies**

- **Explanation**: Use advanced retrieval methods to ensure relevant documents are included.

- **Steps to Follow**:
  - Implement Advanced Retrieval Strategies:
    - Recursive Table Retrieval
    - Document Summary Index
    - Metadata Filters
    - Recursive Retrieval

#### **2. Fine-tune Embeddings**

- **Explanation**: Customize embedding models to better capture the nuances of your data.

### **Example: Embedding Fine-tuning**

```python
finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model_id="BAAI/bge-small-en",
    model_output_path="finetuned_model",
    val_dataset=val_dataset,
)
finetune_engine.finetune()
embed_model = finetune_engine.get_finetuned_model()
```

---

## **Pain Point 4 - Not Extracted**

### **Problem**

- **Description**: The system struggles to extract the correct answer from the provided context.

### **Solutions**

#### **1. Prompt Compression**

- **Explanation**: Use techniques like LongLLMLingua to compress context.

### **Example: LongLLMLingua Implementation**

```python
from llama_index.postprocessor import LongLLMLinguaPostprocessor

node_postprocessor = LongLLMLinguaPostprocessor(
    instruction_str="Given the context, please answer the final question",
    target_token=300,
    rank_method="longllmlingua",
    additional_compress_kwargs={
        "condition_compare": True,
        "context_budget": "+100",
        "reorder_context": "sort",
    },
)

query_engine = index.as_query_engine(node_postprocessors=[node_postprocessor])
response = query_engine.query("Your question here")
```

---

## **Pain Point 5 - Incorrect Format**

### **Problem**

- **Description**: When instructed to provide information in a specific format (e.g., tables, lists), the model may overlook these instructions.

### **Solutions**

#### **1. Prompt Enhancement**

- **Explanation**: Clearly specify desired output format, use keywords, and provide examples to guide the model.

- **Steps to Follow**:
  - **Clear Instructions**: Be explicit about format.
  - **Use Keywords**: Include words like "list", "table", "JSON".
  - **Provide Examples**: Show expected output format.

#### **2. Output Parsing**

- **Explanation**: Employ output parsing modules to enforce specific output formats.

### **Example: Output Parsing Implementation**

```python
from llama_index.output_parsers import LangchainOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

response_schemas = [
    ResponseSchema(
        name="Education",
        description="Describe the author's educational background.",
    ),
    ResponseSchema(
        name="Work",
        description="Describe the author's work experience.",
    ),
]

lc_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
output_parser = LangchainOutputParser(lc_output_parser)

llm = OpenAI(output_parser=output_parser)
query_engine = index.as_query_engine(service_context=ServiceContext.from_defaults(llm=llm))
response = query_engine.query("What are the author's main achievements?")
```

---

## **Pain Point 6 - Incorrect Specificity**

### **Problem**

- **Description**: Model responses may lack necessary detail or specificity, often being too vague or general.

### **Solution**

#### **Advanced Retrieval Strategies**

- **Explanation**: Implement methods that retrieve information at the correct level of granularity.

- **Steps to Follow**:
  - Use Advanced Retrieval Techniques:
    - Small to Large Retrieval
    - Sentence Window Retrieval
    - Recursive Retrieval

---

## **Pain Point 7 - Incomplete**

### **Problem**

- **Description**: The model provides partial answers that don't cover all aspects of the query, even when the information is present.

### **Solution**

#### **Query Transformations**

- **Explanation**: Use query understanding layers to transform queries before retrieval.

### **Example: HyDE Query Transformation**

```python
from llama_index import HyDEQueryTransform, TransformQueryEngine

hyde = HyDEQueryTransform(include_original=True)
query_engine = index.as_query_engine()
query_engine = TransformQueryEngine(query_engine, query_transform=hyde)
response = query_engine.query("What did Paul Graham do after going to RISD?")
```

---

## **Pain Point 8 - Data Ingestion Scalability**

### **Problem**

- **Description**: Challenges in efficiently managing and processing large volumes of data can lead to performance bottlenecks.

### **Solution**

#### **Ingestion Pipeline Parallelization**

- **Explanation**: Use parallel processing to speed up data ingestion.

### **Example: IngestionPipeline Implementation**

```python
from llama_index import IngestionPipeline

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=1024, chunk_overlap=20),
        TitleExtractor(),
        OpenAIEmbedding(),
    ]
)

nodes = pipeline.run(documents=documents, num_workers=4)
```

---

## **Pain Point 9 - Structured Data QA**

### **Problem**

- **Description**: Accurately interpreting user queries to retrieve relevant structured data can be challenging.

### **Solutions**

#### **1. Chain-of-Table Pack**

- **Explanation**: Use the "chain-of-table" method to improve reasoning over tabular data.

#### **2. Mix-Self-Consistency Pack**

- **Explanation**: Combine textual and symbolic reasoning with a self-consistency mechanism.

### **Example: Mix-Self-Consistency Implementation**

```python
query_engine = MixSelfConsistencyQueryEngine(
    df=table,
    llm=llm,
    text_paths=5,
    symbolic_paths=5,
    aggregation_mode="self-consistency",
    verbose=True,
)
response = await query_engine.aquery("Your structured data question")
```

---

## **Pain Point 10 - Complex PDF Data Extraction**

### **Problem**

- **Description**: Extracting data from complex PDFs, such as embedded tables, presents significant challenges.

### **Solution**

#### **Embedded Table Retrieval**

- **Explanation**: Use specialized tools to parse and retrieve data from tables embedded within PDFs.

- **Steps to Follow**:
  - Convert PDFs to HTML: Use tools like pdf2htmlEX.
  - Use EmbeddedTablesUnstructuredRetrieverPack: Process and retrieve data from complex PDFs.

### **Example: EmbeddedTablesUnstructuredRetrieverPack Implementation**

```python
embedded_tables_unstructured_pack = EmbeddedTablesUnstructuredRetrieverPack(
    "data/document.html",
    nodes_save_path="nodes.pkl"
)

response = embedded_tables_unstructured_pack.run("What are the total operating expenses?")
print(response.response)
```

---

## **Pain Point 11 - Backup Model(s)**

### **Problem**

- **Description**: Relying on a single model can lead to failures if the model encounters issues like rate limits or outages.

### **Solutions**

#### **1. Neutrino Router**

- **Explanation**: Implement a collection of models and use intelligent routing to select the best model for each query.

### **Example: Neutrino Implementation**

```python
from llama_index.llms import Neutrino

llm = Neutrino(
    api_key="YOUR_NEUTRINO_API_KEY",
    router="your_router_name"
)

response = llm.complete("Your query here")
print(f"Optimal model used: {response.raw['model']}")
```

#### **2. OpenRouter**

- **Explanation**: Use a unified API that provides access to multiple models with automatic failover capabilities.

### **Example: OpenRouter Implementation**

```python
from llama_index.llms import OpenRouter
from llama_index.llms import ChatMessage

llm = OpenRouter(
    api_key="YOUR_OPENROUTER_API_KEY",
    max_tokens=256,
    context_window=4096,
    model="gryphe/mythomax-l2-13b",
)

message = ChatMessage(role="user", content="Tell me a joke")
response = llm.chat([message])
print(response)
```

## **Pain Point 12 - LLM Security**

### **Problem**

- **Description**: Risks associated with prompt injection, unsafe outputs, and unintended disclosure of sensitive information.

### **Solution**

#### **Llama Guard**

- **Explanation**: Employ tools like Llama Guard to classify and moderate both inputs and outputs, enforcing security policies.

- **Steps to Follow**:
  - Download and Initialize LlamaGuardModeratorPack
  - Define Security Policies
  - Integrate Moderation

### **Example: Llama Guard Implementation**

```python
llamaguard_pack = LlamaGuardModeratorPack(custom_taxonomy=unsafe_categories)

def moderate_and_query(query_engine, query):
    moderator_response_for_input = llamaguard_pack.run(query)
    if moderator_response_for_input == 'safe':
        response = query_engine.query(query)
        moderator_response_for_output = llamaguard_pack.run(str(response))
        if moderator_response_for_output != 'safe':
            response = 'The response is not safe. Please ask a different question.'
    else:
        response = 'This query is not safe. Please ask a different question.'
    return response

final_response = moderate_and_query(query_engine, "Your query here")
print(final_response)
```

---

## **General Techniques for Building Production-Grade RAG**

### **Summary**

Beyond addressing specific pain points, implementing general techniques can improve the performance and robustness of RAG applications.

### **Techniques Covered**:

1. Decouple Chunks Used for Retrieval vs. Chunks Used for Synthesis
2. Structured Retrieval for Larger Document Sets
3. Dynamic Chunk Retrieval Based on Your Task
4. Optimize Context Embeddings

---

## **Pain Point 13 - Decoupling Chunks**

### **Motivation**

- **Problem**: The optimal chunk representation for retrieval may differ from that used for synthesis.

### **Key Techniques**

#### **1. Embed Document Summaries**

- **Explanation**: Embed summaries that link to detailed chunks, allowing high-level retrieval before diving into details.

#### **2. Embed Sentences with Context Windows**

- **Explanation**: Embed sentences and link them to a window around the sentence, ensuring more detailed retrieval with proper context.

---

## **Conclusion**

### **Summary**

By addressing these pain points with the proposed solutions and implementing general techniques, you can build robust, accurate, and scalable RAG applications suitable for production environments.
