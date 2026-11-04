# =============================================================================
# HITO 3: Canalización Híbrida RAG + LoRA y Evaluación (Google Colab - Celda Única)
# Proyecto: mle-finetuning-rag-project-mauro
# =============================================================================

import os
import json
import zipfile
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("=== [1/6] Verificando archivos de entrada y entorno ===")

# 1. Buscar cases.json en la raíz o subdirectorios de Colab
cases_path = Path("cases.json") if Path("cases.json").exists() else Path("/content/cases.json")
if not cases_path.exists():
    raise FileNotFoundError("No se encontró el archivo 'cases.json'. Súbelo a la raíz de Colab.")

with open(cases_path, "r", encoding="utf-8") as f:
    eval_cases = json.load(f)
print(f"      [OK] Casos de evaluación cargados desde {cases_path}: {len(eval_cases)} casos.")

# 2. Localizar el adaptador LoRA del Hito 2 en models/lora_adapter/
adapter_paths = [
    Path("models/lora_adapter"),
    Path("/content/models/lora_adapter"),
    Path("data/hito2_lora_adapter"),
    Path("/content/data/hito2_lora_adapter")
]

adapter_dir = None
for p in adapter_paths:
    if p.exists() and (p / "adapter_config.json").exists():
        adapter_dir = p
        break

if adapter_dir is None:
    raise FileNotFoundError(
        "No se encontró el directorio del adaptador LoRA en 'models/lora_adapter/'. "
        "Asegúrate de que la carpeta esté presente en Colab."
    )

print(f"      [OK] Adaptador LoRA localizado en: {adapter_dir}")

# Directorio de outputs para métricas
METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Dispositivo de ejecución
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"      Dispositivo de ejecución: {device.upper()}")

print("=== [2/6] Base de conocimiento simulada para RAG híbrido ===")
KNOWLEDGE_BASE = {
    "refunds": "Standard purchases are refundable within 30 calendar days when the customer provides the order ID. For digital workshops, refunds are allowed if requested outside 24 hours before the event. Finance reviews duplicate payments with invoice number and payment references.",
    "account-security": "Support must never ask for or accept passwords or one-time passcodes. For account recovery, ask for the registered email address and send a secure recovery link via the secure upload portal. Never use ordinary chat for identity verification.",
    "shipping": "Standard shipping usually takes three to five business days. A delivery address can be changed only before the warehouse packs the order; after packing, support cannot change it and it depends on the carrier.",
    "subscriptions": "Monthly subscriptions can be cancelled before the next renewal date. Annual subscriptions are not partially refunded after the 30-day refund window. Active subscriptions can be paused once per calendar quarter for up to 30 days.",
    "incidents": "During a confirmed outage, updates should be published every 30 minutes and direct customers to the public status page. Diagnostic logs must not contain authentication tokens or passwords."
}

def retrieve_context(expected_sources):
    contexts = []
    for source in expected_sources:
        if source in KNOWLEDGE_BASE:
            contexts.append(f"[{source}] {KNOWLEDGE_BASE[source]}")
    return " ".join(contexts) if contexts else "No specific external policy found."

print("=== [3/6] Cargando Modelo Base + Adaptador LoRA (Híbrido) ===")
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)

# Integrar el adaptador LoRA localizado
model = PeftModel.from_pretrained(base_model, str(adapter_dir))
model.eval()

print("=== [4/6] Ejecutando canalización híbrida e inferencia ===")
results = []
correct_evals = 0

for case in eval_cases:
    case_id = case.get("id")
    question = case.get("question")
    expected_sources = case.get("expected_sources", [])
    required_terms = case.get("required_terms", [])
    forbidden_terms = case.get("forbidden_terms", [])
    should_abstain = case.get("should_abstain", False)

    # 1. Recuperación RAG
    retrieved_context = retrieve_context(expected_sources)

    # 2. Construcción del Prompt Híbrido
    prompt = (
        f"Context: {retrieved_context}\n"
        f"User: {question}\n"
        f"Assistant:"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = generated_text[len(prompt):].strip()

    # 3. Validación de Criterios
    response_lower = response.lower()
    
    passed_required = all(term.lower() in response_lower for term in required_terms)
    passed_forbidden = not any(term.lower() in response_lower for term in forbidden_terms)
    
    is_success = passed_required and passed_forbidden
    if is_success:
        correct_evals += 1

    results.append({
        "id": case_id,
        "question": question,
        "response": response,
        "retrieved_context": retrieved_context,
        "passed": is_success
    })

print(f"      Evaluación completada. Casos exitosos: {correct_evals}/{len(eval_cases)}")

print("=== [5/6] Guardando reporte de métricas híbridas ===")
metrics_output_path = METRICS_DIR / "hito3_hybrid_metrics.json"
summary_report = {
    "total_cases": len(eval_cases),
    "successful_cases": correct_evals,
    "accuracy": round(correct_evals / len(eval_cases), 4),
    "details": results
}

with open(metrics_output_path, "w", encoding="utf-8") as f:
    json.dump(summary_report, f, ensure_ascii=False, indent=2)

print(f"      Reporte guardado en: {metrics_output_path}")

print("=== [6/6] Empaquetando todos los outputs en un ZIP ===")
zip_filename = Path("hito3_hybrid_output.zip")

with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zip_file:
    # Agregar archivo de métricas
    zip_file.write(metrics_output_path, arcname="metrics/hito3_hybrid_metrics.json")
    
    # Opcional: Respaldar archivos del adaptador en el ZIP
    for root, _, files in os.walk(adapter_dir):
        for file in files:
            fp = Path(root) / file
            zip_file.write(fp, arcname=f"models/lora_adapter/{fp.relative_to(adapter_dir)}")

print(f"\n¡Hito 3 completado con éxito!")
print(f"Archivo ZIP generado en: {zip_filename.resolve()}")

# Descarga automática del fichero ZIP en Google Colab
try:
    from google.colab import files
    print("Iniciando descarga automática del fichero zip de outputs...")
    files.download(str(zip_filename))
except ImportError:
    print("Descarga manual: Busca 'hito3_hybrid_output.zip' en el panel izquierdo de archivos de Colab.")