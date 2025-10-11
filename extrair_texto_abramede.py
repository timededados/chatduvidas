import fitz  # PyMuPDF
import json
import os

# ============================
# CONFIGURAÇÕES
# ============================

pdfs = [
    ("ABRAMEDE 0001-0900.pdf", 1),      # PDF 1: páginas 1–900
    ("ABRAMEDE 0901-1800.pdf", 901),    # PDF 2: páginas 901–1800
    ("ABRAMEDE 1801-2412.pdf", 1801)    # PDF 3: páginas 1801–2412
]

output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "abramede_texto.json")

# ============================
# EXTRAÇÃO
# ============================

paginas = []

for arquivo, pagina_inicial in pdfs:
    if not os.path.exists(arquivo):
        print(f"⚠️ Arquivo não encontrado: {arquivo}")
        continue

    print(f"📘 Lendo {arquivo} ...")
    doc = fitz.open(arquivo)

    for i, page in enumerate(doc, start=pagina_inicial):
        texto = page.get_text("text").strip()
        paginas.append({
            "pagina": i,
            "texto": texto
        })
        if i % 100 == 0:
            print(f"  → Extraídas até a página {i}")

    doc.close()

# ============================
# SALVAR ARQUIVO FINAL
# ============================

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(paginas, f, ensure_ascii=False, indent=2)

print(f"\n✅ Extração concluída!")
print(f"Total de páginas: {len(paginas)}")
print(f"Arquivo salvo em: {output_path}")
