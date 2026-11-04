# =============================================================================
# HITO 2: Entrenamiento del Adaptador LoRA y Comprobación de Outputs (Celda Única)
# Proyecto: mle-finetuning-rag-project-mauro
# =============================================================================

# 0. Actualizar o instalar dependencias compatibles requeridas por PEFT
import subprocess
import sys

print("=== [0/7] Actualizando dependencias de entorno (torchao) ===")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "torchao>=0.16.0", "peft", "transformers", "datasets"])

import os
import json
import zipfile
from pathlib import Path
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

print("=== [1/7] Verificando entorno y búsqueda de instruction_examples.jsonl ===")

# Rutas de búsqueda para el archivo subido en la raíz de Colab
possible_paths = [
    Path("instruction_examples.jsonl"),
    Path("/content/instruction_examples.jsonl")
]

train_file_path = None
for path in possible_paths:
    if path.exists():
        train_file_path = path
        break

if train_file_path is None:
    raise FileNotFoundError(
        "No se encontró el archivo 'instruction_examples.jsonl' en la raíz de Colab. "
        "Por favor, asegúrate de haberlo subido al panel de archivos."
    )

print(f"      [OK] Archivo de entrenamiento localizado en: {train_file_path}")

# Directorio donde se guardará el adaptador antes de empaquetarlo
OUTPUT_DIR = Path("models/lora_adapter")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Verificación de aceleración por GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"      Dispositivo de ejecución detectado: {device.upper()}")
if device == "cpu":
    print("      [ADVERTENCIA] Estás ejecutando en CPU. Se recomienda cambiar el entorno a GPU T4 en Colab.")

print("=== [2/7] Carga y formateo de datos JSONL ===")
training_texts = []
with open(train_file_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        
        # Formateo del ejemplo de entrenamiento incluyendo contexto, pregunta y respuesta
        context = record.get("context", "")
        question = record.get("question", "")
        answer = record.get("answer", "")
        
        formatted_prompt = (
            f"Context: {context}\n"
            f"User: {question}\n"
            f"Assistant: {answer}"
        )
        training_texts.append(formatted_prompt)

print(f"      Total de ejemplos de entrenamiento cargados: {len(training_texts)}")

dataset = Dataset.from_dict({"text": training_texts})

print("=== [3/7] Cargando modelo base y tokenizador ===")
MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto" if device == "cuda" else None
)

print("=== [4/7] Configurando Adaptador LoRA (PEFT) ===")
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

print("=== [5/7] Tokenizando el dataset ===")
def tokenize_function(examples):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )
    outputs["labels"] = outputs["input_ids"].copy()
    return outputs

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

print("=== [6/7] Entrenando el adaptador LoRA ===")
training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=2,
    num_train_epochs=3,
    save_strategy="epoch",
    fp16=(device == "cuda"),
    optim="adamw_torch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()

# Guardar pesos y tokenizador en el directorio local
model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print("=== [7/7] Empaquetando los outputs en un archivo ZIP ===")
zip_filename = Path("hito2_lora_adapter.zip")

with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zip_file:
    for root, _, files in os.walk(OUTPUT_DIR):
        for file in files:
            file_path = Path(root) / file
            arcname = file_path.relative_to(OUTPUT_DIR)
            zip_file.write(file_path, arcname=arcname)

print(f"\n¡Hito 2 completado con éxito!")
print(f"El adaptador LoRA se ha guardado y comprimido en: {zip_filename.resolve()}")

# Descarga automática opcional si estás en Google Colab
try:
    from google.colab import files
    print("Iniciando descarga automática del fichero zip...")
    files.download(str(zip_filename))
except ImportError:
    print("Para descargar el archivo manualmente, ubica 'hito2_lora_adapter.zip' en el panel de archivos de Colab.")