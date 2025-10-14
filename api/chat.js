// api/chat.js
import OpenAI from "openai";
import fs from "fs/promises";
import path from "path";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const DATA_DIR = path.join(process.cwd(), "data");
const BOOK_PATH = path.join(DATA_DIR, "abramede_texto.json");
const EMB_PATH = path.join(DATA_DIR, "abramede_embeddings.json");

const EMB_MODEL = "text-embedding-3-small";
const CHAT_MODEL = "gpt-4o-mini";
const TOP_K = 6;
const MAX_CONTEXT_TOKENS = 3000;

// ------------------ Funções auxiliares ------------------
function dot(a, b) {
  return a.reduce((s, v, i) => s + v * b[i], 0);
}
function norm(a) {
  return Math.sqrt(a.reduce((s, x) => s + x * x, 0));
}
function cosineSim(a, b) {
  return dot(a, b) / (norm(a) * norm(b) + 1e-8);
}

// ------------------ Função principal ------------------
export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();

  try {
    const { question } = req.body;
    if (!question || !question.trim())
      return res.status(400).json({ error: "Pergunta vazia" });

    // 1️⃣ Gerar variações da pergunta
    const variationPrompt = `
Você é um assistente que ajuda a gerar variações de consulta de busca em textos técnicos.
Dada a pergunta do usuário, gere até 6 variações curtas (1-12 palavras cada) que mantenham o sentido,
sem adicionar conteúdo novo. Responda apenas em JSON: {"variations": ["..."]}

Pergunta: """${question}"""
`;

    const varResp = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages: [
        { role: "system", content: "Você gera variações de consultas para busca textual sem adicionar conteúdo." },
        { role: "user", content: variationPrompt }
      ],
      temperature: 0.2,
      max_tokens: 300
    });

    const rawVar = varResp.choices?.[0]?.message?.content?.trim() || "";
    let variations = [question];
    try {
      const parsed = JSON.parse(rawVar);
      if (parsed?.variations?.length) variations = parsed.variations;
    } catch {
      variations = rawVar.split(/\r?\n/).filter(Boolean).slice(0, 6);
      if (!variations.length) variations = [question];
    }

    // 2️⃣ Carrega o livro
    const bookRaw = await fs.readFile(BOOK_PATH, "utf8");
    const pages = JSON.parse(bookRaw);
    const pageMap = new Map(pages.map(p => [p.pagina, p.texto]));

    // 3️⃣ Carrega ou gera embeddings
    let pageEmbeddings;
    try {
      const embRaw = await fs.readFile(EMB_PATH, "utf8");
      pageEmbeddings = JSON.parse(embRaw);
    } catch {
      pageEmbeddings = [];
      for (const p of pages) {
        const txt = (p.texto || "").slice(0, 2000);
        const emb = await openai.embeddings.create({
          model: EMB_MODEL,
          input: txt || " "
        });
        pageEmbeddings.push({ pagina: p.pagina, embedding: emb.data[0].embedding });
      }
      await fs.writeFile(EMB_PATH, JSON.stringify(pageEmbeddings), "utf8");
    }

    // 4️⃣ Busca semântica
    const aggregate = new Map();
    for (const query of variations) {
      const qEmb = await openai.embeddings.create({
        model: EMB_MODEL,
        input: query
      });
      const queryEmb = qEmb.data[0].embedding;
      for (const pe of pageEmbeddings) {
        const score = cosineSim(queryEmb, pe.embedding);
        const prev = aggregate.get(pe.pagina) || 0;
        aggregate.set(pe.pagina, prev + score);
      }
    }

    // Ordena páginas mais relevantes
    const sorted = Array.from(aggregate.entries())
      .map(([pagina, score]) => ({ pagina, score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, TOP_K);

    // 🔹 Inclui 1 página anterior e 1 posterior para cada selecionada
    const expandedSet = new Set();
    for (const { pagina } of sorted) {
      expandedSet.add(pagina);
      if (pagina > 1) expandedSet.add(pagina - 1);
      expandedSet.add(pagina + 1);
    }

    // Elimina duplicadas e ordena
    const expandedPages = Array.from(expandedSet).sort((a, b) => a - b);

    const selected = expandedPages.map(pagina => ({
      pagina,
      texto: pageMap.get(pagina) || ""
    }));

    // 5️⃣ Monta o contexto
    let contextBuilder = [];
    let totalLen = 0;
    for (const s of selected) {
      const snippet = (s.texto || "").trim();
      const len = snippet.length;
      if (totalLen + len > MAX_CONTEXT_TOKENS * 4) break;
      contextBuilder.push(`--- Página ${s.pagina} ---\n${snippet}\n`);
      totalLen += len;
    }

    const contextText = contextBuilder.join("\n");
    if (!contextText.trim())
      return res.json({ answer: "Não encontrei conteúdo no livro." });

    // 6️⃣ Prompt para resposta literal e restrita
    const systemInstruction = `
Você é um assistente que responde perguntas exclusivamente com base no texto abaixo.
⚠️ REGRAS IMPORTANTES:
- NÃO adicione informações externas ao texto.
- NÃO use conhecimento médico, técnico ou enciclopédico de fora do livro.
- SÓ utilize frases, trechos ou paráfrases curtas do texto fornecido.
- NÃO preencha lacunas nem interprete significados.
- Se o texto não contiver a resposta, diga exatamente:
  "Não encontrei conteúdo no livro."
- Cite sempre a(s) página(s) usada(s) entre parênteses, ex: (p. 45).
`;

    const userPrompt = `
Conteúdo do livro (trechos das páginas selecionadas):

${contextText}

Pergunta do usuário:
"""${question}"""

Responda de forma literal, usando apenas o texto acima.
`;

    const chatResp = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages: [
        { role: "system", content: systemInstruction },
        { role: "user", content: userPrompt }
      ],
      temperature: 0,
      max_tokens: 800
    });

    const answer =
      chatResp.choices?.[0]?.message?.content?.trim() ||
      "Não encontrei conteúdo no livro.";

    return res.status(200).json({
      answer,
      used_pages: expandedPages
    });

  } catch (err) {
    console.error("Erro no /api/chat:", err);
    return res.status(500).json({ error: String(err) });
  }
}
