"""
Hito 1: Script Unificado de Procesamiento de Datos, Indexación y Evaluación del Recuperador
Proyecto: mle-finetuning-rag-project-mauro
"""

import os
import json
import re
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple

# ---------------------------------------------------------------------------
# Configuración y Rutas del Proyecto
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
TRAINING_DIR = DATA_DIR / "training"
EVAL_DIR = DATA_DIR / "evaluation"
METRICS_DIR = BASE_DIR / "metrics"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DIR.mkdir(parents=True, exist_ok=True)
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Clase para Manejo de Documentos y Chunking
# ---------------------------------------------------------------------------
class DocumentChunk:
    def __init__(self, chunk_id: str, doc_name: str, section: str, content: str):
        self.chunk_id = chunk_id
        self.doc_name = doc_name
        self.section = section
        self.content = content

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_name": self.doc_name,
            "section": self.section,
            "content": self.content
        }

# ---------------------------------------------------------------------------
# Recuperador Léxico / BM25 Simulado para Verificación
# ---------------------------------------------------------------------------
class SimpleBM25Retriever:
    def __init__(self, chunks: List[DocumentChunk], similarity_threshold: float = 0.15):
        self.chunks = chunks
        self.threshold = similarity_threshold

    def _tokenize(self, text: str) -> set:
        words = re.findall(r'\w+', text.lower())
        return set(w for w in words if len(w) > 2)

    def retrieve(self, query: str, top_k: int = 3) -> Tuple[List[DocumentChunk], List[float]]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return [], []

        scores = []
        for chunk in self.chunks:
            chunk_tokens = self._tokenize(chunk.content)
            intersection = query_tokens.intersection(chunk_tokens)
            union = query_tokens.union(chunk_tokens)
            jaccard = len(intersection) / len(union) if union else 0.0
            scores.append((chunk, jaccard))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_results = [(chunk, score) for chunk, score in scores[:top_k] if score >= self.threshold]
        
        retrieved_chunks = [item[0] for item in top_results]
        retrieved_scores = [item[1] for item in top_results]
        return retrieved_chunks, retrieved_scores

# ---------------------------------------------------------------------------
# Funciones de Carga y Evaluación
# ---------------------------------------------------------------------------
def load_and_chunk_knowledge_base(kb_dir: Path) -> List[DocumentChunk]:
    chunks = []
    chunk_counter = 0

    md_files = list(kb_dir.glob("*.md"))
    if not md_files:
        print(f"[WARN] No se encontraron archivos .md en {kb_dir}. Se utilizarán fragmentos de demostración.")
        demo_chunks = [
            DocumentChunk("chunk_001", "northstar_shipping_policy.md", "Envíos Nacionales", 
                          "Northstar ofrece envío estándar gratuito en compras superiores a $50 dólares. El tiempo de entrega estándar es de 3 a 5 días hábiles."),
            DocumentChunk("chunk_002", "northstar_returns_policy.md", "Política de Devoluciones", 
                          "Las devoluciones se aceptan dentro de los 30 días posteriores a la compra. El producto debe estar en su estado original con etiquetas."),
            DocumentChunk("chunk_003", "northstar_warranty.md", "Garantía de Productos", 
                          "Northstar ofrece 1 año de garantía contra defectos de fabricación en todos los productos electrónicos.")
        ]
        return demo_chunks

    for file_path in md_files:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        sections = re.split(r'(^#+\s+.*)', content, flags=re.MULTILINE)
        current_section = "Introducción"
        for part in sections:
            part = part.strip()
            if not part:
                continue
            if part.startswith("#"):
                current_section = part.lstrip("#").strip()
            else:
                chunk_counter += 1
                chunk_id = f"chunk_{chunk_counter:03d}"
                chunks.append(DocumentChunk(chunk_id, file_path.name, current_section, part))

    return chunks

def evaluate_retriever(retriever: SimpleBM25Retriever, eval_cases: List[Dict[str, Any]], k: int = 3) -> Dict[str, float]:
    hit_count = 0
    mrr_sum = 0.0
    recall_sum = 0.0
    abstention_correct = 0
    total_unsupported = 0
    total_supported = 0

    for case in eval_cases:
        query = case.get("query", "")
        expected_docs = case.get("expected_docs", [])
        is_supported = case.get("is_supported", True)

        retrieved_chunks, scores = retriever.retrieve(query, top_k=k)
        retrieved_doc_names = [c.doc_name for c in retrieved_chunks]

        if not is_supported:
            total_unsupported += 1
            if len(retrieved_chunks) == 0:
                abstention_correct += 1
        else:
            total_supported += 1
            # Hit Rate
            hits = [1 for doc in expected_docs if doc in retrieved_doc_names]
            if sum(hits) > 0:
                hit_count += 1

            # Recall@K
            if expected_docs:
                found_count = len(set(expected_docs).intersection(set(retrieved_doc_names)))
                recall_sum += found_count / len(expected_docs)

            # MRR@K
            mrr_rank = 0.0
            for rank, doc in enumerate(retrieved_doc_names, start=1):
                if doc in expected_docs:
                    mrr_rank = 1.0 / rank
                    break
            mrr_sum += mrr_rank

    total_cases = len(eval_cases)
    hit_rate = (hit_count / total_supported) if total_supported > 0 else 0.0
    mrr = (mrr_sum / total_supported) if total_supported > 0 else 0.0
    recall = (recall_sum / total_supported) if total_supported > 0 else 0.0
    abstention_accuracy = (abstention_correct / total_unsupported) if total_unsupported > 0 else 1.0

    return {
        "total_cases": total_cases,
        "total_supported": total_supported,
        "total_unsupported": total_unsupported,
        "hit_rate_k": round(hit_rate, 4),
        "mrr_k": round(mrr, 4),
        "recall_k": round(recall, 4),
        "abstention_accuracy": round(abstention_accuracy, 4)
    }

def generate_html_report(metrics: Dict[str, float], output_path: Path):
    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reporte de Métricas - Hito 1</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f4f7f6;
            color: #333;
            margin: 0;
            padding: 40px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: #ffffff;
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        }}
        h1 {{
            color: #1e293b;
            border-bottom: 3px solid #3b82f6;
            padding-bottom: 10px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 25px;
        }}
        .card {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        .card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #2563eb;
            margin-top: 10px;
        }}
        .card .label {{
            font-size: 0.9em;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 30px;
        }}
        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background-color: #f1f5f9;
            color: #334155;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Métricas de Evaluación - Hito 1 (Recuperador)</h1>
        <p><strong>Proyecto:</strong> Northstar Customer Support Assistant (mle-finetuning-rag-project-mauro)</p>
        
        <div class="metric-grid">
            <div class="card">
                <div class="label">Hit Rate @ K</div>
                <div class="value">{metrics['hit_rate_k'] * 100:.1f}%</div>
            </div>
            <div class="card">
                <div class="label">MRR @ K</div>
                <div class="value">{metrics['mrr_k']:.3f}</div>
            </div>
            <div class="card">
                <div class="label">Recall @ K</div>
                <div class="value">{metrics['recall_k'] * 100:.1f}%</div>
            </div>
            <div class="card">
                <div class="label">Precisión de Abstención</div>
                <div class="value">{metrics['abstention_accuracy'] * 100:.1f}%</div>
            </div>
        </div>

        <h2>Resumen del Dataset de Evaluación</h2>
        <table>
            <tr><th>Métrica</th><th>Valor</th></tr>
            <tr><td>Total de Casos Evaluados</td><td>{metrics['total_cases']}</td></tr>
            <tr><td>Casos Soportados (Con Evidencia)</td><td>{metrics['total_supported']}</td></tr>
            <tr><td>Casos No Soportados (Sin Evidencia)</td><td>{metrics['total_unsupported']}</td></tr>
        </table>
    </div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[INFO] Reporte HTML generado exitosamente en: {output_path}")

# ---------------------------------------------------------------------------
# Función Principal
# ---------------------------------------------------------------------------
def main():
    print("=== Iniciando Ejecución del Hito 1 ===")
    
    # 1. Carga y Chunking
    print("[1/4] Procesando base de conocimiento...")
    chunks = load_and_chunk_knowledge_base(KNOWLEDGE_BASE_DIR)
    print(f"      Total de chunks generados: {len(chunks)}")

    # 2. Indexación
    print("[2/4] Inicializando recuperador vectorial/léxico...")
    retriever = SimpleBM25Retriever(chunks, similarity_threshold=0.1)

    # 3. Datos de Evaluación
    print("[3/4] Cargando casos de prueba de evaluación...")
    demo_eval_cases = [
        {
            "query": "¿Cuál es el costo del envío estándar y cuánto tarda?",
            "expected_docs": ["northstar_shipping_policy.md"],
            "is_supported": True
        },
        {
            "query": "¿Cuántos días tengo para devolver un producto?",
            "expected_docs": ["northstar_returns_policy.md"],
            "is_supported": True
        },
        {
            "query": "¿Cuál es la política de garantía para ropa?",
            "expected_docs": ["northstar_warranty.md"],
            "is_supported": True
        },
        {
            "query": "¿Ofrecen servicio de reparación de bicicletas a domicilio?",
            "expected_docs": [],
            "is_supported": False
        }
    ]

    # 4. Evaluación y Métricas
    print("[4/4] Calculando métricas del recuperador...")
    metrics = evaluate_retriever(retriever, demo_eval_cases, k=3)
    print("      Resultados obtenidos:", metrics)

    # 5. Generación del Reporte HTML
    report_file = METRICS_DIR / "hito1_retrieval_metrics.html"
    generate_html_report(metrics, report_file)
    print("=== Hito 1 Completado Exitosamente ===")

if __name__ == "__main__":
    main()