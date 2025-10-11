import json
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Caminhos dos arquivos
DATA_DIR = "./data"
INPUT_JSON = os.path.join(DATA_DIR, "abramede_texto.json")
OUTPUT_JSON = os.path.join(DATA_DIR, "embeddings_abramede.json")

# Modelo local (rápido e leve)
MODEL_NAME = "all-MiniLM-L6-v2"

print(f"🔹 Carregando modelo local de embeddings: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# Carrega o texto já extraído
print(f"📖 Lendo texto extraído de {INPUT_JSON}")
with open(INPUT_JSON, "r", encoding="utf-8") as f:
    paginas = json.load(f)

# Gera embeddings para cada página
embeddings_data = []
print(f"⚙️ Gerando embeddings para {len(paginas)} páginas...")

for i, pagina in enumerate(tqdm(paginas, desc="Processando")):
    texto = pagina.get("texto", "").strip()
    if not texto:
        continue
    
    # Gera embedding e converte para lista
    embedding = model.encode(texto)
    embedding = embedding.tolist()  # conversão manual
    
    embeddings_data.append({
        "pagina": pagina.get("pagina"),
        "texto": texto,
        "embedding": embedding
    })

# Salva o arquivo final no mesmo formato esperado pelo servidor Node
print(f"💾 Salvando embeddings locais em {OUTPUT_JSON}")
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(embeddings_data, f, ensure_ascii=False, indent=2)

print("✅ Concluído! Embeddings locais prontos para uso.")
