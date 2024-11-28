# **Construyendo Aplicaciones RAG de Grado de Producción: Abordando Puntos de Dolor Comunes**

## **Introducción**

- **Resumen**: Prototipar una aplicación de Recuperación Aumentada con Generación (RAG) es sencillo, pero construir una que sea performante, robusta y escalable para un gran corpus de conocimiento es un desafío.

- **Objetivo**: Esta presentación aborda puntos de dolor comunes al construir aplicaciones RAG de grado de producción y proporciona soluciones detalladas, incluyendo técnicas avanzadas para optimización.

---

## **Punto de Dolor 1 - Contenido Faltante**

### **Problema**

- **Descripción**: El sistema RAG proporciona respuestas plausibles pero incorrectas cuando la respuesta real no está presente en la base de conocimiento. Esto lleva a que los usuarios reciban información engañosa.

### **Soluciones**

#### **1. Limpieza de Datos**

- **Explicación**: Asegúrate de que la fuente de datos sea precisa y esté libre de información conflictiva. La mala calidad de datos conduce a resultados incorrectos.

- **Pasos a Seguir**:
  - **Verificación de Datos**: Audita y actualiza regularmente tu base de conocimiento.
  - **Limpieza de Datos**: Elimina duplicados, corrige errores y resuelve inconsistencias.

#### **2. Mejora de las Indicaciones (Prompts)**

- **Explicación**: Crea indicaciones que alienten al modelo a reconocer cuando no sabe una respuesta.

- **Pasos a Seguir**:
  - **Instrucciones Explícitas**: Utiliza indicaciones como "Si no estás seguro de la respuesta, por favor responde 'No lo sé'."
  - **Ejemplos**: Proporciona ejemplos de indicaciones y respuestas esperadas.

---

## **Punto de Dolor 2 - Documentos Principales Omitidos**

### **Problema**

- **Descripción**: Documentos esenciales no aparecen en los principales resultados devueltos por el componente de recuperación.

### **Soluciones**

#### **1. Ajuste de Hiperparámetros**

- **Explicación**: Ajusta parámetros como `chunk_size` y `similarity_top_k` para mejorar la efectividad de la recuperación.

- **Pasos a Seguir**:
  - **Utiliza `ParamTuner`**: Usa `ParamTuner` de LlamaIndex para automatizar el ajuste.
  - **Experimentación**: Prueba diferentes valores para encontrar las configuraciones óptimas.

### **Ejemplo de Código: Ajuste de Parámetros**

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

#### **2. Reordenamiento (Reranking)**

- **Explicación**: Implementa estrategias de reordenamiento para reorganizar los resultados de recuperación antes de enviarlos al modelo de lenguaje.

- **Pasos a Seguir**:
  - **Aumenta similarity_top_k**: Recupera más documentos para proporcionar una base más amplia para el reordenamiento.
  - **Usa Reordenadores**: Integra herramientas como CohereRerank.

### **Ejemplo: Implementación de CohereRerank**

```python
from llama_index.postprocessor.cohere_rerank import CohereRerank

cohere_rerank = CohereRerank(api_key='TU_COHERE_API_KEY', top_n=2)
query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[cohere_rerank],
)
response = query_engine.query("Tu pregunta aquí")
```

---

## **Punto de Dolor 3 - No en Contexto**

### **Problema**

- **Descripción**: Los documentos recuperados que contienen la respuesta no llegan al contexto utilizado para generar la respuesta, debido a limitaciones en el proceso de consolidación.

### **Soluciones**

#### **1. Ajusta las Estrategias de Recuperación**

- **Explicación**: Utiliza métodos avanzados de recuperación para asegurar que los documentos relevantes estén incluidos.

- **Pasos a Seguir**:
  - Implementa Estrategias Avanzadas de Recuperación:
    - Recuperación Recursiva de Tablas
    - Índice de Resumen de Documentos
    - Filtros de Metadatos
    - Recuperación Recursiva

#### **2. Afinar Embeddings**

- **Explicación**: Personaliza los modelos de embeddings para capturar mejor las sutilezas de tus datos.

### **Ejemplo: Afinación de Embeddings**

```python
finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model_id="BAAI/bge-small-en",
    model_output_path="modelo_afinado",
    val_dataset=val_dataset,
)
finetune_engine.finetune()
embed_model = finetune_engine.get_finetuned_model()
```

---

## **Punto de Dolor 4 - No Extraído**

### **Problema**

- **Descripción**: El sistema tiene dificultades para extraer la respuesta correcta del contexto proporcionado.

### **Soluciones**

#### **1. Compresión de Indicaciones (Prompts)**

- **Explicación**: Utiliza técnicas como LongLLMLingua para comprimir el contexto.

### **Ejemplo: Implementación de LongLLMLingua**

```python
from llama_index.postprocessor import LongLLMLinguaPostprocessor

node_postprocessor = LongLLMLinguaPostprocessor(
    instruction_str="Dado el contexto, por favor responde la pregunta final",
    target_token=300,
    rank_method="longllmlingua",
    additional_compress_kwargs={
        "condition_compare": True,
        "context_budget": "+100",
        "reorder_context": "sort",
    },
)

query_engine = index.as_query_engine(node_postprocessors=[node_postprocessor])
response = query_engine.query("Tu pregunta aquí")
```

---

## **Punto de Dolor 5 - Formato Incorrecto**

### **Problema**

- **Descripción**: Cuando se instruye proporcionar información en un formato específico (por ejemplo, tablas, listas), el modelo puede pasar por alto estas instrucciones.

### **Soluciones**

#### **1. Mejora de las Indicaciones (Prompts)**

- **Explicación**: Especifica claramente el formato de salida deseado, utiliza palabras clave y proporciona ejemplos para guiar al modelo.

- **Pasos a Seguir**:
  - **Instrucciones Claras**: Sé explícito sobre el formato.
  - **Usa Palabras Clave**: Incluye palabras como "lista", "tabla", "JSON".
  - **Proporciona Ejemplos**: Muestra el formato de salida esperado.

#### **2. Análisis de Salida (Output Parsing)**

- **Explicación**: Emplea módulos de análisis de salida para hacer cumplir formatos de salida específicos.

### **Ejemplo: Implementación de Output Parsing**

```python
from llama_index.output_parsers import LangchainOutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

response_schemas = [
    ResponseSchema(
        name="Educación",
        description="Describe la formación educativa del autor.",
    ),
    ResponseSchema(
        name="Trabajo",
        description="Describe la experiencia laboral del autor.",
    ),
]

lc_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
output_parser = LangchainOutputParser(lc_output_parser)

llm = OpenAI(output_parser=output_parser)
query_engine = index.as_query_engine(service_context=ServiceContext.from_defaults(llm=llm))
response = query_engine.query("¿Cuáles son los principales logros del autor?")
```

#### **3. Programas Pydantic**

- **Explicación**: Usa programas Pydantic para definir esquemas de salida estructurados.

### **Ejemplo: Implementación de Programas Pydantic**

```python
from pydantic import BaseModel
from llama_index.program import OpenAIPydanticProgram

class Canción(BaseModel):
    título: str
    duración_segundos: int

class Álbum(BaseModel):
    nombre: str
    artista: str
    canciones: List[Canción]

programa = OpenAIPydanticProgram.from_defaults(
    output_cls=Álbum,
    prompt_template_str="Genera un álbum de ejemplo con canciones inspiradas en {nombre_película}.",
)
output = programa(nombre_película="Inception")
```

---

## **Punto de Dolor 6 - Especificidad Incorrecta**

### **Problema**

- **Descripción**: Las respuestas del modelo pueden carecer de detalle o especificidad necesaria, siendo a menudo demasiado vagas o generales.

### **Solución**

#### **Estrategias Avanzadas de Recuperación**

- **Explicación**: Implementa métodos que recuperen información al nivel correcto de granularidad.

- **Pasos a Seguir**:
  - Usa Técnicas Avanzadas de Recuperación:
    - Recuperación de Pequeño a Grande
    - Recuperación de Ventana de Oración
    - Recuperación Recursiva

---

## **Diapositiva 8: Punto de Dolor 7 - Incompleto**

### **Problema**

- **Descripción**: El modelo proporciona respuestas parciales que no cubren todos los aspectos de la consulta, incluso cuando la información está presente.

### **Solución**

#### **Transformaciones de Consulta**

- **Explicación**: Utiliza capas de comprensión de consulta para transformar las consultas antes de la recuperación.

### **Ejemplo: Transformación de Consulta HyDE**

```python
from llama_index import HyDEQueryTransform, TransformQueryEngine

hyde = HyDEQueryTransform(include_original=True)
query_engine = index.as_query_engine()
query_engine = TransformQueryEngine(query_engine, query_transform=hyde)
response = query_engine.query("¿Qué hizo Paul Graham después de ir a RISD?")
```

---

## **Punto de Dolor 8 - Escalabilidad en la Ingesta de Datos**

### **Problema**

- **Descripción**: Los desafíos para gestionar y procesar eficientemente grandes volúmenes de datos pueden conducir a cuellos de botella en el rendimiento.

### **Solución**

#### **Paralelización del Pipeline de Ingesta**

- **Explicación**: Utiliza procesamiento paralelo para acelerar la ingesta de datos.

### **Ejemplo: Implementación de IngestionPipeline**

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

## **Punto de Dolor 9 - QA de Datos Estructurados**

### **Problema**

- **Descripción**: Interpretar con precisión consultas de usuarios para recuperar datos estructurados relevantes puede ser un desafío.

### **Soluciones**

#### **1. Chain-of-Table Pack**

- **Explicación**: Utiliza el método "chain-of-table" para mejorar el razonamiento sobre datos tabulares.

#### **2. Mix-Self-Consistency Pack**

- **Explicación**: Combina razonamiento textual y simbólico con un mecanismo de autoconsistencia.

### **Ejemplo: Implementación de Mix-Self-Consistency**

```python
query_engine = MixSelfConsistencyQueryEngine(
    df=table,
    llm=llm,
    text_paths=5,
    symbolic_paths=5,
    aggregation_mode="self-consistency",
    verbose=True,
)
response = await query_engine.aquery("Tu pregunta sobre datos estructurados")
```

## **Punto de Dolor 10 - Extracción de Datos de PDFs Complejos**

### **Problema**

- **Descripción**: Extraer datos de PDFs complejos, como tablas incrustadas, presenta desafíos significativos.

### **Solución**

#### **Recuperación de Tablas Incrustadas**

- **Explicación**: Utiliza herramientas especializadas para analizar y recuperar datos de tablas incrustadas dentro de PDFs.

- **Pasos a Seguir**:
  - Convierte PDFs a HTML: Utiliza herramientas como pdf2htmlEX.
  - Usa EmbeddedTablesUnstructuredRetrieverPack: Procesa y recupera datos de PDFs complejos.

### **Ejemplo: Implementación de EmbeddedTablesUnstructuredRetrieverPack**

```python
embedded_tables_unstructured_pack = EmbeddedTablesUnstructuredRetrieverPack(
    "data/documento.html",
    nodes_save_path="nodos.pkl"
)

response = embedded_tables_unstructured_pack.run("¿Cuáles son los gastos operativos totales?")
print(response.response)
```

---

## **Punto de Dolor 11 - Modelo(s) de Respaldo**

### **Problema**

- **Descripción**: Confiar en un solo modelo puede llevar a fallos si el modelo encuentra problemas como límites de tasa o interrupciones.

### **Soluciones**

#### **1. Neutrino Router**

- **Explicación**: Implementa una colección de modelos y utiliza enrutamiento inteligente para seleccionar el mejor modelo para cada consulta.

### **Ejemplo: Implementación de Neutrino**

```python
from llama_index.llms import Neutrino

llm = Neutrino(
    api_key="TU_NEUTRINO_API_KEY",
    router="nombre_de_tu_router"
)

response = llm.complete("Tu consulta aquí")
print(f"Modelo óptimo utilizado: {response.raw['model']}")
```

#### **2. OpenRouter**

- **Explicación**: Usa una API unificada que proporciona acceso a múltiples modelos con capacidades de failover automático.

### **Ejemplo: Implementación de OpenRouter**

```python
from llama_index.llms import OpenRouter
from llama_index.llms import ChatMessage

llm = OpenRouter(
    api_key="TU_OPENROUTER_API_KEY",
    max_tokens=256,
    context_window=4096,
    model="gryphe/mythomax-l2-13b",
)

message = ChatMessage(role="user", content="Cuéntame un chiste")
response = llm.chat([message])
print(response)
```

## **Punto de Dolor 12 - Seguridad en LLM**

### **Problema**

- **Descripción**: Riesgos asociados con la inyección de indicaciones, salidas inseguras y divulgación no intencionada de información sensible.

### **Solución**

#### **Llama Guard**

- **Explicación**: Emplea herramientas como Llama Guard para clasificar y moderar tanto entradas como salidas, haciendo cumplir políticas de seguridad.

- **Pasos a Seguir**:
  - Descarga e Inicializa LlamaGuardModeratorPack
  - Define Políticas de Seguridad
  - Integra Moderación

### **Ejemplo: Implementación de Llama Guard**

```python
llamaguard_pack = LlamaGuardModeratorPack(custom_taxonomy=unsafe_categories)

def moderate_and_query(query_engine, query):
    moderator_response_for_input = llamaguard_pack.run(query)
    if moderator_response_for_input == 'safe':
        response = query_engine.query(query)
        moderator_response_for_output = llamaguard_pack.run(str(response))
        if moderator_response_for_output != 'safe':
            response = 'La respuesta no es segura. Por favor, haz una pregunta diferente.'
    else:
        response = 'Esta consulta no es segura. Por favor, haz una pregunta diferente.'
    return response

final_response = moderate_and_query(query_engine, "Tu consulta aquí")
print(final_response)
```

---

## **Técnicas Generales para Construir RAG de Grado de Producción**

### **Resumen**

Más allá de abordar puntos de dolor específicos, implementar técnicas generales puede mejorar el rendimiento y la robustez de las aplicaciones RAG.

### **Técnicas Cubiertas**:

1. Desacoplar Chunks Usados para Recuperación vs. Chunks Usados para Síntesis
2. Recuperación Estructurada para Conjuntos de Documentos Más Grandes
3. Recuperación Dinámica de Chunks Según tu Tarea
4. Optimizar Embeddings de Contexto

---

## **Punto de Dolor 13 - Desacoplar Chunks**

### **Motivación**

- **Problema**: La representación óptima de chunks para recuperación puede diferir de la utilizada para síntesis.

### **Técnicas Clave**

#### **1. Embebe Resúmenes de Documentos**

- **Explicación**: Embebe resúmenes que enlazan a chunks detallados, permitiendo una recuperación de alto nivel antes de profundizar en detalles.

#### **2. Embebe Oraciones con Ventanas de Contexto**

- **Explicación**: Embebe oraciones y enlázalas a una ventana alrededor de la oración, asegurando una recuperación más detallada con contexto adecuado.

---

## **Conclusión**

### **Resumen**

Al abordar estos puntos de dolor con las soluciones propuestas e implementar técnicas generales, puedes construir aplicaciones RAG robustas, precisas y escalables, adecuadas para entornos de producción.

### **Próximos Pasos**:

1. Explora la Documentación de LlamaIndex
2. Mejora Continua: Prueba, itera y refina tu pipeline RAG
3. Mantente actualizado con las últimas técnicas y mejores prácticas
