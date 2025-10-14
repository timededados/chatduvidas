/**
 * gerar_sumario.js
 * Gera um sumário temático (tópicos e subtópicos) com base no livro e embeddings já existentes.
 * Roda apenas 1x, gera /data/sumario.json
 */

import fs from "fs/promises";
import path from "path";
import OpenAI from "openai";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const DATA_DIR = path.join(process.cwd(), "data");
const BOOK_PATH = path.join(DATA_DIR, "abramede_texto.json");
const EMB_PATH = path.join(DATA_DIR, "abramede_embeddings.json");
const OUTPUT_PATH = path.join(DATA_DIR, "sumario.json");

const EMB_MODEL = "text-embedding-3-small";
const CLUSTER_SIZE = 10; // agrupa 10 páginas por bloco (ajuste conforme desejar)
const CHAT_MODEL = "gpt-4o-mini";

function dot(a, b) {
  return a.reduce((s, v, i) => s + v * b[i], 0);
}
function norm(a) {
  return Math.sqrt(a.reduce((s, x) => s + x * x, 0));
}
function cosineSim(a, b) {
  return dot(a, b) / (norm(a) * norm(b) + 1e-8);
}

async function main() {
  console.log("📘 Lendo arquivo do livro e embeddings...");
  const bookRaw = await fs.readFile(BOOK_PATH, "utf8");
  const pages = JSON.parse(bookRaw);

  const embRaw = await fs.readFile(EMB_PATH, "utf8");
  const embeddings = JSON.parse(embRaw);

  if (pages.length !== embeddings.length)
    console.warn("⚠️ Quantidade de embeddings e páginas não bate exatamente.");

  // Monta blocos de páginas consecutivas (para contexto mais longo)
  const blocks = [];
  for (let i = 0; i < pages.length; i += CLUSTER_SIZE) {
    const subset = pages.slice(i, i + CLUSTER_SIZE);
    const blockText = subset.map(p => `Página ${p.pagina}:\n${p.texto}`).join("\n\n");
    const startPage = subset[0].pagina;
    const endPage = subset[subset.length - 1].pagina;
    blocks.push({ startPage, endPage, text: blockText });
  }

  console.log(`📚 Criados ${blocks.length} blocos de ${CLUSTER_SIZE} páginas.`);

  const sumario = [];

  for (const [i, block] of blocks.entries()) {
    console.log(`🧠 Processando bloco ${i + 1}/${blocks.length} (p. ${block.startPage}-${block.endPage})...`);

    const prompt = `
Você é um assistente que cria sumários hierárquicos a partir de um texto técnico.
Leia o conteúdo fornecido e identifique:
- Os tópicos principais abordados
- Os subtópicos dentro de cada tópico
- As páginas que cobrem cada tópico ou subtópico (com base nas páginas informadas)

Retorne o resultado em JSON no formato:
[
  {
    "topico": "Título principal",
    "paginas": [n, n+1, ...],
    "subtopicos": [
      {"titulo": "Subtítulo 1", "paginas": [n, n+1]},
      {"titulo": "Subtítulo 2", "paginas": [n]}
    ]
  }
]

Não invente títulos fora do conteúdo. Use linguagem técnica conforme o texto.
Conteúdo (páginas ${block.startPage}-${block.endPage}):

${block.text}
`;

    const resp = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages: [
        { role: "system", content: "Você é um analista de textos médicos que gera sumários hierárquicos em JSON." },
        { role: "user", content: prompt }
      ],
      temperature: 0.2,
      max_tokens: 4000
    });

    const raw = resp.choices[0].message?.content?.trim();
    try {
      const parsed = JSON.parse(raw.match(/\[[\s\S]*\]/)?.[0] || "[]");
      sumario.push(...parsed);
    } catch (err) {
      console.warn(`⚠️ Erro ao parsear JSON do bloco ${i + 1}:`, err.message);
      console.log("Resposta recebida:", raw);
    }
  }

  await fs.writeFile(OUTPUT_PATH, JSON.stringify(sumario, null, 2), "utf8");
  console.log(`✅ Sumário gerado com sucesso em ${OUTPUT_PATH}`);
}

main().catch(err => console.error("❌ Erro:", err));
