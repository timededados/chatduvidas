// api/chat.js
import OpenAI from "openai";
import fs from "fs/promises";
import path from "path";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const DATA_DIR = path.join(process.cwd(), "data");
const BOOK_PATH = path.join(DATA_DIR, "abramede_texto.json");
const EMB_PATH = path.join(DATA_DIR, "abramede_embeddings.json");
const SUM_PATH = path.join(DATA_DIR, "sumario_final.json");

const EMB_MODEL = "text-embedding-3-small";
const CHAT_MODEL = "gpt-4o-mini";
const TOP_K = 6;
const MAX_CONTEXT_TOKENS = 3000;

// ---------- Funções auxiliares ----------
function dot(a, b) {
  return a.reduce((s, v, i) => s + v * b[i], 0);
}
function norm(a) {
  return Math.sqrt(a.reduce((s, x) => s + x * x, 0));
}
function cosineSim(a, b) {
  return dot(a, b) / (norm(a) * norm(b) + 1e-8);
}

// Determinismo e normalização simples
function seedFromString(s) {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h) + s.charCodeAt(i);
  return Math.abs(h >>> 0);
}
function normalizeStr(s) {
  return (s || "")
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "");
}
function countOccurrences(text, token) {
  if (!token) return 0;
  const re = new RegExp(`\\b${token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "g");
  return (text.match(re) || []).length;
}

// Busca simples no sumário por similaridade textual
function searchSummary(sumario, query) {
  const normalized = query.toLowerCase();
  const results = [];
  for (const top of sumario) {
    const allText = `${top.topico} ${top.subtopicos
      ?.map(s => s.titulo)
      .join(" ")}`.toLowerCase();
    if (allText.includes(normalized)) {
      results.push(...top.paginas);
      for (const s of top.subtopicos || []) {
        results.push(...(s.paginas || []));
      }
    }
  }
  return Array.from(new Set(results)).sort((a, b) => a - b);
}

// ---------- Função principal ----------
export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();

  try {
    const { question } = req.body;
    if (!question || !question.trim())
      return res.status(400).json({ error: "Pergunta vazia" });

    // 1️⃣ Remover geração de variações estocásticas (multi-query) para evitar flutuação
    const variations = [question]; // consulta única e determinística

    // 2️⃣ Carrega dados (inalterado)
    const [bookRaw, embRaw, sumRaw] = await Promise.all([
      fs.readFile(BOOK_PATH, "utf8"),
      fs.readFile(EMB_PATH, "utf8"),
      fs.readFile(SUM_PATH, "utf8")
    ]);
    const pages = JSON.parse(bookRaw);
    const pageEmbeddings = JSON.parse(embRaw);
    const sumario = JSON.parse(sumRaw);

    const pageMap = new Map(pages.map(p => [p.pagina, p.texto]));

    // 3️⃣ Busca no sumário (inalterado)
    const pagesFromSummary = searchSummary(sumario, question);

    // 4️⃣ Consulta de embedding única e ranking híbrido (embedding + lexical + boost sumário)
    const qEmbResp = await openai.embeddings.create({
      model: EMB_MODEL,
      input: question
    });
    const queryEmb = qEmbResp.data[0].embedding;

    const qNorm = normalizeStr(question);
    const qTokens = Array.from(
      new Set(qNorm.split(/\W+/).filter(t => t && t.length > 2))
    );

    // Calcular scores por página
    let minEmb = Infinity, maxEmb = -Infinity, maxLex = 0;
    const prelim = [];
    for (const pe of pageEmbeddings) {
      const embScore = cosineSim(queryEmb, pe.embedding);
      const raw = pageMap.get(pe.pagina) || "";
      const txt = normalizeStr(raw);
      let lexScore = 0;
      for (const t of qTokens) lexScore += countOccurrences(txt, t);
      prelim.push({
        pagina: pe.pagina,
        embScore,
        lexScore,
        inSummary: pagesFromSummary.includes(pe.pagina)
      });
      if (embScore < minEmb) minEmb = embScore;
      if (embScore > maxEmb) maxEmb = embScore;
      if (lexScore > maxLex) maxLex = lexScore;
    }

    const ranked = prelim.map(r => {
      const embNorm = (r.embScore - minEmb) / (Math.max(1e-8, maxEmb - minEmb));
      const lexNorm = maxLex > 0 ? r.lexScore / maxLex : 0;
      const summaryBoost = r.inSummary ? 0.08 : 0;
      const finalScore = 0.7 * embNorm + 0.3 * lexNorm + summaryBoost;
      return { ...r, embNorm, lexNorm, finalScore };
    })
    .sort((a, b) => {
      if (b.finalScore !== a.finalScore) return b.finalScore - a.finalScore;
      return a.pagina - b.pagina; // desempate determinístico
    });

    if (!ranked.length) {
      return res.json({ answer: "Não encontrei conteúdo no livro.", used_pages: [] });
    }

    // 5️⃣ Seleciona 1 única página para contexto (consistência de citação)
    const bestPage = ranked[0].pagina;
    const selectedPages = [bestPage];

    // 6️⃣ Monta o contexto só da melhor página
    const snippet = (pageMap.get(bestPage) || "").trim();
    if (!snippet) {
      return res.json({ answer: "Não encontrei conteúdo no livro.", used_pages: [] });
    }
    const contextText = `--- Página ${bestPage} ---\n${snippet}\n`;

    // 7️⃣ Prompt restritivo: citar exatamente 1 página e trecho literal
    const systemInstruction = `
Você responde exclusivamente com base no texto abaixo.
Regras:
- Não adicione informações externas.
- Use somente frases originais do texto fornecido.
- Cite exatamente 1 página e inclua pelo menos 1 trecho literal entre aspas dessa página.
- Se a informação não estiver no texto, responda exatamente: "Não encontrei conteúdo no livro."
`.trim();

    const userPrompt = `
Conteúdo do livro (apenas 1 página):
${contextText}

Pergunta do usuário:
"""${question}"""

Responda usando SOMENTE palavras do texto, cite a página ${bestPage} e inclua trechos literais entre aspas.
Se a página não contiver a resposta, diga: "Não encontrei conteúdo no livro."
`.trim();

    // 8️⃣ Geração determinística
    const seed = seedFromString(question);
    const chatResp = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages: [
        { role: "system", content: systemInstruction },
        { role: "user", content: userPrompt }
      ],
      temperature: 0,
      top_p: 1,
      frequency_penalty: 0,
      presence_penalty: 0,
      n: 1,
      max_tokens: 900,
      seed
    });

    const answer =
      chatResp.choices?.[0]?.message?.content?.trim() ||
      "Não encontrei conteúdo no livro.";

    return res.status(200).json({
      answer,
      used_pages: selectedPages
    });

  } catch (err) {
    console.error("Erro no /api/chat:", err);
    return res.status(500).json({ error: String(err) });
  }
}
