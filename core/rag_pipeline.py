#!/usr/bin/env python3
"""
Production-Ready RAG (Retrieval-Augmented Generation) Pipeline
for Industrial Maintenance Diagnostics Agent.

Stack:
- Configuration & Validation: Pydantic (v2)
- Vector Store: ChromaDB
- Embeddings: Sentence-Transformers ('all-MiniLM-L6-v2')
- LLM Inference: Ollama ('llama3.2')
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import uuid
from typing import Any, Dict, List, Optional

# Core third-party dependencies
from pydantic import BaseModel, Field, field_validator

try:
    import chromadb
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
except ImportError:
    chromadb = None
    Documents = Any
    EmbeddingFunction = object
    Embeddings = Any

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    import ollama
except ImportError:
    ollama = None

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("RAGPipeline")


# =====================================================================
# 1. Pydantic Configuration Model
# =====================================================================
class RAGConfig(BaseModel):
    """
    Configuration model for the Industrial Maintenance RAG pipeline.
    Validates hyperparameters, paths, and model specifications.
    """
    chunk_size: int = Field(
        default=300,
        ge=50,
        description="Maximum character length per document chunk."
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Character overlap between consecutive chunks to preserve context."
    )
    collection_name: str = Field(
        default="industrial_maintenance_docs",
        description="ChromaDB collection identifier for maintenance documents."
    )
    db_path: str = Field(
        default="./chroma_db",
        description="Filesystem directory path for persistent ChromaDB storage."
    )
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="HuggingFace model name used for sentence-transformers embeddings."
    )
    llm_model_name: str = Field(
        default="llama3.2",
        description="Target Ollama model tag for LLM inference."
    )
    ollama_host: str = Field(
        default="http://localhost:11434",
        description="URL host of the Ollama inference server."
    )
    top_k: int = Field(
        default=3,
        ge=1,
        description="Number of most relevant chunks to retrieve for context generation."
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, overlap: int, info) -> int:
        chunk_size = info.data.get("chunk_size", 300)
        if overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({overlap}) must be strictly less than chunk_size ({chunk_size})"
            )
        return overlap


# =====================================================================
# 2. Text Chunking Strategy
# =====================================================================
def chunk_text(text: str, chunk_size: int = 300, chunk_overlap: int = 50) -> List[str]:
    """
    Splits text into chunks of maximum `chunk_size` characters with `chunk_overlap`.
    Preserves word boundaries by splitting at whitespace whenever possible.

    Args:
        text: Raw document text to be chunked.
        chunk_size: Maximum character length of each chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of non-empty text chunks.
    """
    clean_text = text.strip()
    if not clean_text:
        return []
    if len(clean_text) <= chunk_size:
        return [clean_text]

    chunks: List[str] = []
    start = 0
    total_len = len(clean_text)

    while start < total_len:
        end = min(start + chunk_size, total_len)

        # Attempt to break cleanly on whitespace if not reaching the end
        if end < total_len:
            split_idx = clean_text.rfind(" ", start, end)
            # Ensure split point is past the overlap window to guarantee forward progress
            if split_idx != -1 and split_idx > start + chunk_overlap:
                chunk = clean_text[start:split_idx].strip()
                if chunk:
                    chunks.append(chunk)
                start = split_idx + 1 - chunk_overlap
                continue

        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= total_len:
            break
        start += (chunk_size - chunk_overlap)

    return chunks


# =====================================================================
# 3. Context Builder (Required Format)
# =====================================================================
# Reference to default initialized pipeline instance for automatic retrieval
_GLOBAL_PIPELINE: Optional["IndustrialRAGPipeline"] = None


def build_agent_context(query: str, docs: Optional[List[str]] = None) -> str:
    """
    Constructs the exact agent context prompt required for industrial maintenance.

    If `docs` is omitted or None, automatically retrieves relevant documentation
    from the active IndustrialRAGPipeline instance.

    Args:
        query: The user maintenance query or diagnostic question.
        docs: List of retrieved document strings.

    Returns:
        The exact formatted context string.
    """
    if docs is None:
        if _GLOBAL_PIPELINE is not None:
            docs = _GLOBAL_PIPELINE.retrieve(query)
        else:
            docs = []

    return f"""SYSTEM: You are an expert industrial maintenance agent. 
    Use the provided documents to answer the query accurately. 
    If the answer is not in the documents, state that you do not know.
    DOCUMENTS:{docs}
    QUERY: {query}
   """


# =====================================================================
# 4. Sentence-Transformers Embedding Function for ChromaDB
# =====================================================================
class LocalSentenceTransformerEmbedding(EmbeddingFunction):
    """
    ChromaDB compatible embedding adapter using sentence-transformers 'all-MiniLM-L6-v2'.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required. Install via: pip install sentence-transformers"
            )
        logger.info("Loading embedding model: %s ...", model_name)
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully.")

    def __call__(self, input):
        """
        Embeds a list of text documents into a list of vector floats.
        """
        if not input:
            return []
        
        embeddings = self.model.encode(
            list(input),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()


# =====================================================================
# 5. Production Industrial RAG Pipeline
# =====================================================================
class IndustrialRAGPipeline:
    """
    Production-ready RAG pipeline integrating ChromaDB vector storage,
    SentenceTransformers embeddings, and Ollama LLM inference.
    """
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        global _GLOBAL_PIPELINE
        _GLOBAL_PIPELINE = self

        # Initialize Embedding Function
        try:
            self.embedding_fn = LocalSentenceTransformerEmbedding(
                model_name=self.config.embedding_model_name
            )
        except Exception as exc:
            logger.error("Failed to load embedding model '%s': %s", self.config.embedding_model_name, exc)
            raise

        # Initialize ChromaDB with robust error handling
        self._init_chromadb()

        # Initialize Ollama client with connection validation
        self._init_ollama()

        # Seed default maintenance docs if collection is empty
        if self.collection.count() == 0:
            self.seed_default_maintenance_docs()

    def _init_chromadb(self) -> None:
        """
        Initializes persistent ChromaDB storage with fallback to in-memory on error.
        """
        if chromadb is None:
            raise ImportError("chromadb is required. Install via: pip install chromadb")

        try:
            os.makedirs(self.config.db_path, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=self.config.db_path)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.config.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"description": "Industrial maintenance operational documents"},
            )
            logger.info(
                "ChromaDB persistent store initialized at '%s'. Collection '%s' contains %d items.",
                self.config.db_path,
                self.config.collection_name,
                self.collection.count(),
            )
        except Exception as exc:
            logger.error(
                "Failed to initialize ChromaDB persistent client at '%s': %s",
                self.config.db_path,
                exc,
            )
            try:
                logger.warning("Attempting fallback to ephemeral in-memory ChromaDB client...")
                self.chroma_client = chromadb.EphemeralClient()
                self.collection = self.chroma_client.get_or_create_collection(
                    name=self.config.collection_name,
                    embedding_function=self.embedding_fn,
                )
                logger.info("ChromaDB in-memory fallback successfully initialized.")
            except Exception as fallback_exc:
                logger.critical("Fatal: ChromaDB initialization failed completely: %s", fallback_exc)
                raise RuntimeError(f"ChromaDB initialization failed: {exc}") from exc

    def _init_ollama(self) -> None:
        """
        Initializes Ollama client and tests daemon connectivity.
        """
        self.ollama_connected = False
        if ollama is None:
            logger.warning("ollama package is not installed. Install via: pip install ollama")
            self.ollama_client = None
            return

        try:
            self.ollama_client = ollama.Client(host=self.config.ollama_host)
            # Probe connection and available models
            models_response = self.ollama_client.list()
            self.ollama_connected = True

            available_models = [
                m.get("name", "") if isinstance(m, dict) else getattr(m, "model", "")
                for m in models_response.get("models", [])
            ]
            logger.info("Ollama connection verified at %s.", self.config.ollama_host)

            # Check if specified model is present
            model_base = self.config.llm_model_name.split(":")[0]
            matched = any(model_base in m for m in available_models)
            if not matched:
                logger.warning(
                    "Model '%s' not detected in Ollama model list. You can pull it using: ollama pull %s",
                    self.config.llm_model_name,
                    self.config.llm_model_name,
                )
        except Exception as exc:
            logger.warning(
                "Ollama daemon is not reachable at %s (%s). "
                "RAG retrieval will function, but LLM generation will be unavailable until Ollama is started. "
                "To start: 'ollama serve' and 'ollama pull %s'.",
                self.config.ollama_host,
                exc,
                self.config.llm_model_name,
            )
            self.ollama_connected = False

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Chunks and indexes a single document into the vector store.

        Args:
            doc_id: Unique document identifier (e.g. manual ID or SOP number).
            text: Full text content of the document.
            metadata: Optional dictionary of metadata attributes.

        Returns:
            List of generated chunk IDs.
        """
        chunks = chunk_text(
            text=text,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        if not chunks:
            logger.warning("Document '%s' produced 0 chunks. Skipping indexing.", doc_id)
            return []

        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        chunk_metadatas = [
            {
                **(metadata or {}),
                "doc_id": doc_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ]

        self.collection.add(
            ids=chunk_ids,
            documents=chunks,
            metadatas=chunk_metadatas,
        )
        logger.info(
            "Indexed document '%s' into %d chunks in collection '%s'.",
            doc_id,
            len(chunks),
            self.config.collection_name,
        )
        return chunk_ids

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """
        Retrieves the top-k most relevant document chunks for a query.

        Args:
            query: The user query or problem description.
            top_k: Optional override for number of chunks to retrieve.

        Returns:
            List of retrieved chunk text strings.
        """
        k = top_k if top_k is not None else self.config.top_k
        if self.collection.count() == 0:
            logger.warning("ChromaDB collection '%s' is empty.", self.config.collection_name)
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(k, self.collection.count()),
            )
            docs = results.get("documents", [[]])
            return docs[0] if docs else []
        except Exception as exc:
            logger.error("ChromaDB query failed for '%s': %s", query, exc)
            return []

    def generate(self, prompt: str) -> str:
        """
        Sends the compiled context prompt to the Ollama LLM for generation.

        Args:
            prompt: Formatted prompt containing system instructions, query, and docs.

        Returns:
            The generated response string from the model.
        """
        if not self.ollama_connected or self.ollama_client is None:
            return (
                f"[Ollama Offline] Connection to Ollama server at {self.config.ollama_host} is currently inactive. "
                f"Please run 'ollama serve' and ensure model '{self.config.llm_model_name}' is downloaded "
                f"('ollama pull {self.config.llm_model_name}') to enable generative inference."
            )

        try:
            response = self.ollama_client.chat(
                model=self.config.llm_model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            if isinstance(response, dict):
                return response.get("message", {}).get("content", "")
            return getattr(getattr(response, "message", None), "content", "") or ""
        except Exception as exc:
            logger.error("Inference call failed with model '%s': %s", self.config.llm_model_name, exc)
            return f"[Generation Error] Failed to generate response from Ollama: {exc}"

    def query(self, user_query: str) -> Dict[str, Any]:
        """
        Executes an end-to-end RAG workflow:
        1. Retrieve relevant documentation chunks from ChromaDB.
        2. Build formatted agent context via `build_agent_context`.
        3. Query Ollama LLM with the context prompt.

        Args:
            user_query: The technical maintenance query.

        Returns:
            Dictionary containing the query, retrieved docs, context, and answer.
        """
        retrieved_docs = self.retrieve(user_query)
        context_prompt = build_agent_context(query=user_query, docs=retrieved_docs)
        answer = self.generate(context_prompt)

        return {
            "query": user_query,
            "retrieved_docs": retrieved_docs,
            "context_prompt": context_prompt,
            "response": answer,
        }

    def add_documents(self,documents: List[Dict[str, Any]],) -> List[str]:

        """
        Batch adds multiple documents to the vector store.
        Each item in `documents` should be a dict with at least 'doc_id' and 'text'.
        """

        all_chunk_ids: List[str] = []
        for doc in documents:
            doc_id = doc.get("doc_id") or doc.get("id") or str(uuid.uuid4())
            text = doc.get("text") or doc.get("content") or ""
            metadata = doc.get("metadata", {})
            chunk_ids = self.add_document(doc_id=doc_id, text=text, metadata=metadata)
            all_chunk_ids.extend(chunk_ids)
        return all_chunk_ids

    def retrieve_docs(self, query: str, top_k: Optional[int] = None) -> List[str]:
        """Alias for retrieve()."""
        return self.retrieve(query=query, top_k=top_k)

    def build_agent_context(self, query: str, docs: Optional[List[str]] = None) -> str:
        """Instance method to build the agent context for a query."""
        if docs is None:
            docs = self.retrieve(query)
        return build_agent_context(query=query, docs=docs)

    def ask(self, user_query: str) -> str:
        """Convenience method returning the generated answer string directly."""
        result = self.query(user_query)
        return result["response"]

    def seed_default_maintenance_docs(self) -> None:
        """
        Seeds standard industrial maintenance documentation into ChromaDB
        if the collection is currently empty.
        """
        if self.collection.count() > 0:
            logger.info(
                "Collection '%s' already contains %d documents. Skipping default seeding.",
                self.config.collection_name,
                self.collection.count(),
            )
            return

        sample_docs = [
            {
                "doc_id": "SOP-HYD-401",
                "metadata": {"system": "Hydraulics", "type": "SOP", "criticality": "HIGH"},
                "text": (
                    "HYDRAULIC PUMP CAVITATION PROTOCOL (SOP-HYD-401): "
                    "Cavitation occurs when local fluid pressure drops below the vapor pressure, forming vapor bubbles "
                    "that violently implode against pump internal surfaces. "
                    "Key Symptoms: Loud metallic rattling noise (sounds like marbles or gravel inside casing), "
                    "high-frequency casing vibration (spikes above 2.5 kHz), erratic discharge pressure fluctuations, "
                    "and rapid temperature elevation at the pump head. "
                    "Immediate Corrective Actions: 1. Immediately verify suction shut-off valve is '100%' locked open. "
                    "2. Inspect suction line strainer/filter for clogging or collapsed mesh; clean or replace filter element. "
                    "3. Check hydraulic reservoir oil level and fluid temperature (normal operating range: 40°C to 55°C, max 65°C). "
                    "4. Verify oil viscosity complies with ISO VG 46/68; aerated or deteriorated oil must be flushed. "
                    "5. Bleed trapped air from the highest vent plug on the pump casing until a steady bubble-free stream appears."
                ),
            },
            {
                "doc_id": "MAN-BRG-202",
                "metadata": {"system": "Mechanical", "type": "Manual", "criticality": "CRITICAL"},
                "text": (
                    "ROLLING ELEMENT BEARING VIBRATION & THERMAL DIAGNOSTICS (MAN-BRG-202): "
                    "Permissible operating temperature for deep-groove ball and spherical roller bearings is under 80°C (176°F). "
                    "Temperatures exceeding 90°C require immediate load reduction and lubrication verification. "
                    "Vibration Thresholds: According to ISO 10816-3 for Class II industrial machines (15kW to 75kW): "
                    "Velocity RMS below 2.8 mm/s is normal (Zone A/B). "
                    "Velocity RMS between 2.8 mm/s and 4.5 mm/s indicates restricted operation / pending maintenance (Zone C). "
                    "Velocity RMS exceeding 4.5 mm/s represents unacceptable danger (Zone D) requiring immediate shutdown. "
                    "Lubrication: Use lithium-complex synthetic grease (NLGI Grade 2). Re-greasing interval is 2,000 operational hours "
                    "or every 6 months. Do not over-grease; purge old grease through relief port."
                ),
            },
            {
                "doc_id": "STD-MOT-105",
                "metadata": {"system": "Electrical", "type": "Standard", "criticality": "HIGH"},
                "text": (
                    "THREE-PHASE INDUCTION MOTOR ELECTRICAL FAULT STANDARD (STD-MOT-105): "
                    "Winding Insulation Testing: Measure insulation resistance using a calibrated megohmmeter (Megger) at 1,000V DC. "
                    "Minimum acceptable insulation resistance is 5 Megaohms at 40°C. Readings below 1.0 Megaohm indicate severe "
                    "moisture ingress or dielectric breakdown; do not energize motor. "
                    "Phase Current Balance: Maximum allowable current unbalance between phases is 5%. "
                    "Formula: % Unbalance = (Max Deviation from Average / Average Current) * 100. "
                    "A voltage imbalance of 1% produces approximately 6% to 10% current imbalance and up to 25% motor temperature rise."
                ),
            },
            {
                "doc_id": "GUIDE-CMP-303",
                "metadata": {"system": "Pneumatics", "type": "Guide", "criticality": "MEDIUM"},
                "text": (
                    "ROTARY SCREW AIR COMPRESSOR PREVENTIVE MAINTENANCE (GUIDE-CMP-303): "
                    "Air/Oil Separator Differential Pressure: Monitor separator differential pressure gauge weekly. "
                    "Replace oil separator cartridge when differential pressure exceeds 0.8 bar (11.6 psi) or every 4,000 running hours. "
                    "Condensate Drainage: Test zero-loss electronic condensate drain valves daily. Defective drains lead to water "
                    "contamination in plant air headers and corrosion of downstream pneumatic actuators. "
                    "Operating Temperature: Nominal discharge temperature is 82°C to 93°C. Temperatures below 70°C promote condensation "
                    "inside the oil reservoir, while temperatures over 105°C trigger high-temperature emergency shutdown."
                ),
            },
        ]

        logger.info("Seeding %d industrial maintenance manuals into ChromaDB...", len(sample_docs))
        for doc in sample_docs:
            self.add_document(
                doc_id=doc["doc_id"],
                text=doc["text"],
                metadata=doc["metadata"],
            )
        logger.info("Seeding complete. Vector store ready.")


# =====================================================================
# Aliases for flexible import conventions
# =====================================================================
RAGPipeline = IndustrialRAGPipeline
RAGPipelineConfig = RAGConfig


# =====================================================================
# 6. Dataset Ingestion Helpers
# =====================================================================
def load_and_transform_ai4i_dataset(file_path: str) -> str:
    """
    Reads the AI4I 2020 CSV and converts each row into a natural language 
    document so the Vector DB and LLM can understand it.
    """
    try:
        import pandas as pd
    except ImportError:
        logger.warning("pandas is required to load the AI4I dataset.")
        return ""

    df = pd.read_csv(file_path)
    documents = ""
    for _, row in df.iterrows():
        doc_text = (
            f"Machine Record ID {row['UDI']}: "
            f"Product ID {row['Product ID']}, Type {row['Type']}. "
            f"Air temperature was {row['Air temperature [K]']}K, "
            f"Process temperature was {row['Process temperature [K]']}K. "
            f"Rotational speed was {row['Rotational speed [rpm]']} rpm, "
            f"Torque was {row['Torque [Nm]']} Nm, "
            f"Tool wear was {row['Tool wear [min]']} minutes. "
            f"Machine failure occurred: {row['Machine failure']}. "
        )
        documents += doc_text + "\n"

    return documents

