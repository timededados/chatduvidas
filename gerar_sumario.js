/**
 * gerar_sumario.js
 * Gera um sumário temático (tópicos e subtópicos) com base no livro e embeddings já existentes.
 * Roda apenas 1x, gera /data/sumario_final.json em estrutura hierárquica:
 * Seção -> Categorias (capítulos do sumário original) -> Tópicos (com subtópicos)
 */

import fs from "fs/promises";
import path from "path";
import OpenAI from "openai";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const DATA_DIR = path.join(process.cwd(), "data");
const BOOK_PATH = path.join(DATA_DIR, "abramede_texto.json");
const EMB_PATH = path.join(DATA_DIR, "abramede_embeddings.json");
const OUTPUT_PATH = path.join(DATA_DIR, "sumario_final.json"); // ajuste: escrever sumario_final.json
const ORIGINAL_SUMMARY_PATH = path.join(DATA_DIR, "sumario_original.json"); // sumário (TOC) original

const EMB_MODEL = "text-embedding-3-small";
const CLUSTER_SIZE = 10; // agrupa 10 páginas por bloco (ajuste conforme desejar)
const CHAT_MODEL = "gpt-4o-mini";

// Utils de texto e páginas
function normalize(s) {
  return (s || "")
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^\p{L}\p{N}\s\-]/gu, "")
    .replace(/\s+/g, " ")
    .trim();
}
function uniqueSorted(arr = []) {
  return Array.from(new Set(arr.filter(n => Number.isFinite(n)))).sort((a, b) => a - b);
}
function unionPages(...lists) {
  return uniqueSorted(lists.flat());
}
// Similaridade simples por interseção de tokens
function tokenSim(a, b) {
  const A = new Set(normalize(a).split(" ").filter(Boolean));
  const B = new Set(normalize(b).split(" ").filter(Boolean));
  if (!A.size || !B.size) return 0;
  let inter = 0;
  for (const t of A) if (B.has(t)) inter++;
  // média harmônica favorece match mais “estreito”
  const precision = inter / A.size;
  const recall = inter / B.size;
  const denom = precision + recall || 1e-8;
  return (2 * precision * recall) / denom;
}

// Parser do sumário original (TOC) -> { [secao]: { categorias: Set<string> } }
async function extractSectionsAndCategories() {
  const raw = await fs.readFile(ORIGINAL_SUMMARY_PATH, "utf8");
  const pages = JSON.parse(raw);

  const sections = {};
  let currentSection = null;

  const isSectionHeader = line => /^SEÇÃO\s+[IVXLCDM]+/i.test(line);
  const isCoordinator = line => /Coordenador(?:es|a|as)?/i.test(line);
  const isProbablyAuthors = line => line.includes(",") && line.split(",").length >= 2;
  const isNumbering = line => /^\d+\.\s*$/.test(line);
  const isNonEmpty = line => line && line.trim().length > 0;

  for (const p of pages) {
    const lines = (p.texto || "").split("\n").map(s => s.trim()).filter(isNonEmpty);
    for (const line of lines) {
      if (isNumbering(line)) continue;
      if (isSectionHeader(line)) {
        currentSection = line;
        if (!sections[currentSection]) sections[currentSection] = { categorias: new Set() };
        continue;
      }
      if (!currentSection) continue;
      if (isCoordinator(line)) continue;
      // Heurística: o título de capítulo (categoria) costuma ser uma linha sem vírgulas (sem autores)
      // e não é “Conteúdo complementar”, “Prefácio”, etc.
      if (!isProbablyAuthors(line)) {
        // filtrar ruídos óbvios
        if (/^(Sumário|Prefácio|Apresentação|Conteúdo complementar)$/i.test(line)) continue;
        // muitas entradas úteis são de 1 a ~8 palavras
        const wc = line.split(/\s+/).length;
        if (wc >= 1 && wc <= 12) {
          sections[currentSection].categorias.add(line);
        }
      }
    }
  }

  // converter Sets em arrays
  for (const k of Object.keys(sections)) {
    sections[k].categorias = Array.from(sections[k].categorias);
  }
  return sections;
}

// Mescla tópicos duplicados pelo mesmo título normalizado
function mergeDuplicateTopics(items) {
  const map = new Map();
  for (const it of items) {
    const key = normalize(it.topico || it.titulo || "");
    if (!key) continue;
    if (!map.has(key)) {
      map.set(key, {
        topico: it.topico || it.titulo,
        paginas: uniqueSorted(it.paginas || []),
        subtopicos: Array.isArray(it.subtopicos) ? it.subtopicos.slice() : [],
      });
    } else {
      const acc = map.get(key);
      acc.paginas = unionPages(acc.paginas, it.paginas || []);
      if (Array.isArray(it.subtopicos)) acc.subtopicos.push(...it.subtopicos);
    }
  }
  // propagar páginas dos subtópicos para o tópico pai
  for (const [, acc] of map) {
    const subPages = unionPages(...(acc.subtopicos || []).map(st => st.paginas || []));
    acc.paginas = unionPages(acc.paginas, subPages);
    // consolidar subtópicos duplicados por título
    const subMap = new Map();
    for (const st of acc.subtopicos || []) {
      const k = normalize(st.titulo || "");
      if (!k) continue;
      if (!subMap.has(k)) subMap.set(k, { titulo: st.titulo, paginas: uniqueSorted(st.paginas || []) });
      else {
        const sacc = subMap.get(k);
        sacc.paginas = unionPages(sacc.paginas, st.paginas || []);
      }
    }
    acc.subtopicos = Array.from(subMap.values());
  }
  return Array.from(map.values());
}

// Atribui cada tópico a uma (Seção, Categoria) do sumário original
function assignTopicsToSections(mergedTopics, sectionsIndex) {
  const result = [];
  const sectionEntries = Object.entries(sectionsIndex); // [secao, {categorias: []}]

  for (const [secao, meta] of sectionEntries) {
    result.push({ secao, categorias: [] });
  }

  function bestMatchCategory(title) {
    let best = { secao: null, categoria: null, score: 0 };
    for (const [secao, meta] of sectionEntries) {
      for (const cat of meta.categorias) {
        const s = Math.max(
          tokenSim(title, cat),
          tokenSim(cat, title),
          // inclui também substring simples para não perder matches curtos
          normalize(title).includes(normalize(cat)) ? 0.9 : 0,
          normalize(cat).includes(normalize(title)) ? 0.75 : 0
        );
        if (s > best.score) best = { secao, categoria: cat, score: s };
      }
    }
    return best.score >= 0.35 ? best : null; // limiar conservador
  }

  // índice auxiliar para montar categorias
  const secCatMap = new Map(); // key `${secao}||${categoria}` -> { categoria, topicos: [] }
  for (const topic of mergedTopics) {
    const match = bestMatchCategory(topic.topico || "");
    const secao = match?.secao || "SEÇÃO – Miscelânea";
    const categoria = match?.categoria || "Outros tópicos";
    const key = `${secao}||${categoria}`;
    if (!secCatMap.has(key)) {
      secCatMap.set(key, { categoria, paginas: [], topicos: [] });
    }
    secCatMap.get(key).topicos.push(topic);
  }

  // preencher estrutura final por seção
  const bySecao = new Map();
  for (const [key, catObj] of secCatMap) {
    const [secao] = key.split("||");
    if (!bySecao.has(secao)) bySecao.set(secao, []);
    // páginas da categoria = união das páginas dos tópicos filhos
    const catPages = unionPages(...catObj.topicos.map(t => t.paginas || []));
    bySecao.get(secao).push({ categoria: catObj.categoria, paginas: catPages, topicos: catObj.topicos });
  }

  // ordenar categorias e tópicos por primeira página
  function firstPage(arr = []) { return uniqueSorted(arr)[0] ?? Number.MAX_SAFE_INTEGER; }

  for (const sec of result) {
    const cats = bySecao.get(sec.secao) || [];
    for (const c of cats) {
      c.topicos.sort((a, b) => firstPage(a.paginas) - firstPage(b.paginas));
    }
    cats.sort((a, b) => firstPage(a.paginas) - firstPage(b.paginas));
    sec.categorias = cats;
  }

  // incluir seções que não tiveram match (vazias) com array vazio de categorias
  const seen = new Set(result.map(r => r.secao));
  for (const [secao] of sectionEntries) {
    if (!seen.has(secao)) result.push({ secao, categorias: [] });
  }

  // mover “Miscelânea” para o final
  return result.sort((a, b) => {
    const am = /Miscelanea|Miscelânea/i.test(a.secao);
    const bm = /Miscelanea|Miscelânea/i.test(b.secao);
    if (am && !bm) return 1;
    if (!am && bm) return -1;
    return a.secao.localeCompare(b.secao);
  });
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

  // NOVO: pós-processamento e agrupamento por Seção/Categoria do sumário original
  console.log("🧩 Lendo e extraindo Seções/Categorias do sumário original...");
  const sectionsIndex = await extractSectionsAndCategories();

  console.log("🧼 Mesclando tópicos duplicados e propagando páginas de subtópicos...");
  const merged = mergeDuplicateTopics(sumario);

  console.log("📂 Atribuindo tópicos às Seções/Categorias e montando estrutura final (3 níveis)...");
  const structured = assignTopicsToSections(merged, sectionsIndex);

  await fs.writeFile(OUTPUT_PATH, JSON.stringify(structured, null, 2), "utf8");
  console.log(`✅ Sumário gerado com sucesso em ${OUTPUT_PATH}`);
}

main().catch(err => console.error("❌ Erro:", err));
