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

// Recuperação
const RECALL_TOP_K = 60;       // top páginas por similaridade (recall alto)
const TOPIC_PAGES_TOP_K = 8;   // páginas do tópico que entram no contexto
const MAX_CONTEXT_CHARS = 120_000; // ~30k tokens aprox (ajuste conforme necessário)

// ---------- Utils ----------
function dot(a, b) { return a.reduce((s, v, i) => s + v * b[i], 0); }
function norm(a) { return Math.sqrt(a.reduce((s, x) => s + x * x, 0)); }
function cosineSim(a, b) { return dot(a, b) / (norm(a) * norm(b) + 1e-8); }

function normalizeStr(s) {
  return (s || "").toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
}
function tokenize(s) {
  return Array.from(new Set(normalizeStr(s).split(/\W+/).filter(t => t && t.length > 2)));
}
function countOccurrences(text, token) {
  if (!token) return 0;
  const re = new RegExp(`\\b${token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "g");
  return (text.match(re) || []).length;
}
function stableSortByScoreThenPage(arr, scoreKey = "score") {
  return arr.sort((a, b) => {
    if (b[scoreKey] !== a[scoreKey]) return b[scoreKey] - a[scoreKey];
    return a.pagina - b.pagina;
  });
}

// Constrói a lista de tópicos a partir do sumário
function buildTopics(sumario) {
  const topics = [];
  for (const node of sumario) {
    const title = String(node.topico || "").trim();
    const pages = Array.isArray(node.paginas) ? Array.from(new Set(node.paginas)).sort((a, b) => a - b) : [];
    if (title && pages.length) {
      topics.push({ title, pages });
    }
    // subtopicos geralmente não têm páginas; ignorar quando vazio
  }
  return topics;
}

// Seleciona o melhor tópico combinando (1) páginas recuperadas por embeddings e (2) boost lexical do título
function selectBestTopic({ topics, topPages, questionTokens }) {
  let best = null;
  for (const t of topics) {
    // soma dos scores das páginas do tópico presentes em topPages
    let sumScore = 0;
    for (const p of topPages) {
      if (t.pages.includes(p.pagina)) sumScore += p.score;
    }
    // boost lexical pelo título
    const titleNorm = normalizeStr(t.title);
    let lexBoost = 0;
    for (const tok of questionTokens) lexBoost += countOccurrences(titleNorm, tok);
    // peso pequeno no boost lexical
    const finalScore = sumScore + 0.08 * lexBoost;
    if (!best || finalScore > best.finalScore || (finalScore === best.finalScore && t.pages[0] < best.topic.pages[0])) {
      best = { topic: t, finalScore };
    }
  }
  return best?.topic || null;
}

// Gera contexto a partir de páginas selecionadas
function buildContextFromPages({ pagesList, pageMap, maxChars }) {
  let context = "";
  let used = [];
  let acc = 0;
  for (const p of pagesList) {
    const txt = (pageMap.get(p) || "").trim();
    if (!txt) continue;
    const chunk = `--- Página ${p} ---\n${txt}\n\n`;
    if (acc + chunk.length > maxChars) break;
    context += chunk;
    used.push(p);
    acc += chunk.length;
  }
  return { context, usedPages: used };
}

// Validação: quotes devem existir no texto da página indicada
function validateCitations(citations, pageMap, allowedPagesSet) {
  if (!Array.isArray(citations)) return [];
  const valids = [];
  for (const c of citations) {
    const page = Number(c?.page);
    const quote = String(c?.quote || "");
    if (!page || !quote || !allowedPagesSet.has(page)) continue;
    const pageText = pageMap.get(page) || "";
    if (pageText.includes(quote)) {
      valids.push({ page, quote });
    }
    if (valids.length >= 2) break; // limitar a 2 citações
  }
  return valids;
}

// ---------- Handler ----------
export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();

  try {
    const { question } = req.body || {};
    if (!question || !String(question).trim()) {
      return res.status(400).json({ error: "Pergunta vazia" });
    }

    // 1) Carregar dados
    const [bookRaw, embRaw, sumRaw] = await Promise.allSettled([
      fs.readFile(BOOK_PATH, "utf8"),
      fs.readFile(EMB_PATH, "utf8"),
      fs.readFile(SUM_PATH, "utf8")
    ]);

    if (bookRaw.status !== "fulfilled" || embRaw.status !== "fulfilled" || sumRaw.status !== "fulfilled") {
      return res.status(500).json({ error: "Falha ao carregar dados" });
    }

    const pages = JSON.parse(bookRaw.value);              // esperado: [{ pagina: number, texto: string }, ...]
    const pageEmbeddings = JSON.parse(embRaw.value);      // esperado: [{ pagina: number, embedding: number[] }, ...]
    const sumario = JSON.parse(sumRaw.value);             // sumário estruturado

    const pageMap = new Map(pages.map(p => [p.pagina, p.texto || ""]));
    const availablePages = new Set(pages.map(p => p.pagina));

    // 2) Embedding da pergunta
    const qEmbResp = await openai.embeddings.create({
      model: EMB_MODEL,
      input: question
    });
    const queryEmb = qEmbResp.data[0].embedding;

    // 3) Scoring por embeddings e recall de páginas
    const scored = [];
    for (const pe of pageEmbeddings) {
      if (!availablePages.has(pe.pagina)) continue; // ignorar páginas sem texto
      const score = cosineSim(queryEmb, pe.embedding);
      scored.push({ pagina: pe.pagina, score });
    }
    stableSortByScoreThenPage(scored, "score");
    const recallPages = scored.slice(0, RECALL_TOP_K);

    // 4) Seleção de tópico (sumario) com ranking híbrido
    const topics = buildTopics(sumario);
    const qTokens = tokenize(question);
    const bestTopic = selectBestTopic({ topics, topPages: recallPages, questionTokens: qTokens });

    if (!bestTopic) {
      return res.status(200).json({ answer: "Não encontrei conteúdo no livro.", citations: [], used_pages: [] });
    }

    // 5) Dentro do tópico, selecionar páginas com maior score
    const topicPageSet = new Set(bestTopic.pages.filter(p => availablePages.has(p)));
    if (topicPageSet.size === 0) {
      return res.status(200).json({ answer: "Não encontrei conteúdo no livro.", citations: [], used_pages: [] });
    }

    const topicScored = recallPages
      .filter(p => topicPageSet.has(p.pagina))
      .slice(0, RECALL_TOP_K);

    // Se recall não trouxe páginas do tópico, avalie diretamente todas as páginas do tópico
    let finalCandidates = topicScored;
    if (!finalCandidates.length) {
      const rescored = [];
      for (const pe of pageEmbeddings) {
        if (!topicPageSet.has(pe.pagina)) continue;
        rescored.push({ pagina: pe.pagina, score: cosineSim(queryEmb, pe.embedding) });
      }
      stableSortByScoreThenPage(rescored, "score");
      finalCandidates = rescored.slice(0, TOPIC_PAGES_TOP_K);
    }

    // Top páginas do tópico para contexto
    stableSortByScoreThenPage(finalCandidates, "score");
    const selectedPages = finalCandidates
      .slice(0, TOPIC_PAGES_TOP_K)
      .map(x => x.pagina)
      .sort((a, b) => a - b);

    // 6) Montar contexto
    const { context, usedPages } = buildContextFromPages({
      pagesList: selectedPages,
      pageMap,
      maxChars: MAX_CONTEXT_CHARS
    });

    if (!context.trim()) {
      return res.status(200).json({ answer: "Não encontrei conteúdo no livro.", citations: [], used_pages: [] });
    }

    // 7) Prompt: saída estruturada e citações obrigatórias (quotes literais)
    const systemInstruction = `
Você é um assistente que responde exclusivamente com base no conteúdo fornecido.
Regras:
- Não use conhecimento externo.
- Selecione no máximo 2 trechos literais do texto e cite a(s) página(s).
- Se não houver resposta no texto, responda exatamente: "Não encontrei conteúdo no livro."
- Responda no formato JSON: {"answer":"...", "citations":[{"page":<n>,"quote":"..."}]}
`.trim();

    const userPrompt = `
Conteúdo do livro (páginas do tópico: ${bestTopic.title}):

${context}

Pergunta do usuário:
"""${question}"""
`.trim();

    const chatResp = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages: [
        { role: "system", content: systemInstruction },
        { role: "user", content: userPrompt }
      ],
      temperature: 0,
      top_p: 1,
      max_tokens: 900,
      response_format: { type: "json_object" }
    });

    let parsed;
    try {
      const raw = chatResp.choices?.[0]?.message?.content || "";
      parsed = JSON.parse(raw);
    } catch {
      return res.status(200).json({ answer: "Não encontrei conteúdo no livro.", citations: [], used_pages: usedPages, topic: bestTopic.title });
    }

    const answer = String(parsed?.answer || "").trim();
    const citations = Array.isArray(parsed?.citations) ? parsed.citations : [];
    const allowedPagesSet = new Set(usedPages);
    const validCitations = validateCitations(citations, pageMap, allowedPagesSet);

    if (!answer || (validCitations.length === 0 && answer !== "Não encontrei conteúdo no livro.")) {
      return res.status(200).json({ answer: "Não encontrei conteúdo no livro.", citations: [], used_pages: usedPages, topic: bestTopic.title });
    }

    return res.status(200).json({
      answer,
      citations: validCitations,
      used_pages: usedPages,
      topic: bestTopic.title
    });

  } catch (err) {
    console.error("Erro no /api/chat:", err);
    return res.status(500).json({ error: String(err) });
  }
}
