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

// Helpers para seleção de tópico
function tokenize(str) {
  return Array.from(new Set(normalizeStr(str).split(/\W+/).filter(t => t && t.length > 2)));
}
function topicPages(entry) {
  const arr = [...(entry.paginas || [])];
  for (const s of entry.subtopicos || []) arr.push(...(s.paginas || []));
  return Array.from(new Set(arr)).sort((a, b) => a - b);
}
function topicText(entry) {
  const subs = (entry.subtopicos || []).map(s => s.titulo || "").join(" ");
  return `${entry.topico || ""} ${subs}`.trim();
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

    // Mapa rápido página → score
    const pageById = new Map(ranked.map(r => [r.pagina, r]));
    const embByPage = new Map(prelim.map(r => [r.pagina, r.embScore]));

    // 5️⃣ Seleciona 1 tópico do sumário (híbrido lexical + agregação de embeddings das páginas do tópico)
    const qTokensSet = new Set(qTokens);
    const topicScored = [];
    for (const entry of sumario) {
      const tPages = topicPages(entry);
      if (!tPages.length) continue;

      const tText = topicText(entry);
      const tTokens = tokenize(tText);
      let lexOverlap = 0;
      for (const t of tTokens) if (qTokensSet.has(t)) lexOverlap++;

      // soma dos top-5 embScores das páginas do tópico (evita viés por tópicos muito longos)
      const embScores = tPages.map(p => embByPage.get(p) ?? -1).filter(v => v >= 0).sort((a, b) => b - a);
      const embAgg = embScores.slice(0, 5).reduce((s, v) => s + v, 0);

      topicScored.push({ entry, pages: tPages, lexOverlap, embAgg });
    }

    let selectedTopic = null;
    if (topicScored.length) {
      const minLex = Math.min(...topicScored.map(t => t.lexOverlap));
      const maxLex = Math.max(...topicScored.map(t => t.lexOverlap));
      const minEmbAgg = Math.min(...topicScored.map(t => t.embAgg));
      const maxEmbAgg = Math.max(...topicScored.map(t => t.embAgg));

      const scored = topicScored.map(t => {
        const lexNorm = maxLex > minLex ? (t.lexOverlap - minLex) / (maxLex - minLex || 1) : (t.lexOverlap > 0 ? 1 : 0);
        const embNorm = maxEmbAgg > minEmbAgg ? (t.embAgg - minEmbAgg) / (maxEmbAgg - minEmbAgg || 1) : 0;
        const finalScore = 0.4 * lexNorm + 0.6 * embNorm;
        return { ...t, lexNorm, embNorm, finalScore };
      }).sort((a, b) => {
        if (b.finalScore !== a.finalScore) return b.finalScore - a.finalScore;
        // desempate determinístico: menor primeira página
        return (a.pages[0] || Infinity) - (b.pages[0] || Infinity);
      });

      selectedTopic = scored[0];
    }

    // 6️⃣ Monta o contexto com páginas do tópico escolhido (ou fallback para melhores páginas globais)
    let includedPages = [];
    let contextBuilder = [];
    let totalLen = 0;

    if (selectedTopic && selectedTopic.pages.length) {
      // Ordenar páginas do tópico por relevância (score da página) e empatar por número da página
      const orderedTopicPages = [...selectedTopic.pages].sort((a, b) => {
        const sa = pageById.get(a)?.finalScore ?? -1;
        const sb = pageById.get(b)?.finalScore ?? -1;
        if (sb !== sa) return sb - sa;
        return a - b;
      });

      for (const pagina of orderedTopicPages) {
        const snippet = (pageMap.get(pagina) || "").trim();
        if (!snippet) continue;
        const len = snippet.length;
        if (totalLen + len > MAX_CONTEXT_TOKENS * 4) break;
        contextBuilder.push(`--- Página ${pagina} ---\n${snippet}\n`);
        totalLen += len;
        includedPages.push(pagina);
      }
    }

    // Fallback: se nada foi incluído do tópico, use as top páginas globais
    if (!includedPages.length) {
      for (const r of ranked.slice(0, TOP_K)) {
        const snippet = (pageMap.get(r.pagina) || "").trim();
        if (!snippet) continue;
        const len = snippet.length;
        if (totalLen + len > MAX_CONTEXT_TOKENS * 4) break;
        contextBuilder.push(`--- Página ${r.pagina} ---\n${snippet}\n`);
        totalLen += len;
        includedPages.push(r.pagina);
      }
    }

    const contextText = contextBuilder.join("\n");
    if (!contextText.trim()) {
      return res.json({ answer: "Não encontrei conteúdo no livro.", used_pages: [] });
    }

    // 7️⃣ Prompt restritivo: permitir múltiplas páginas e exigir citação literal com página
    const systemInstruction = `
Você responde exclusivamente com base nos trechos abaixo.
Regras:
- Não adicione informações externas.
- Use somente frases originais do texto fornecido.
- Inclua pelo menos 1 citação literal entre aspas com a página de origem. Se usar mais de um trecho, cite a página após cada citação.
- Se a informação não estiver no texto, responda exatamente: "Não encontrei conteúdo no livro."
`.trim();

    const userPrompt = `
Conteúdo do livro (páginas selecionadas):
${contextText}

Pergunta do usuário:
"""${question}"""

Responda SOMENTE com base nesses trechos. Dê uma resposta objetiva e apresente as citações com a(s) página(s) correspondente(s).
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
      used_pages: includedPages
    });

  } catch (err) {
    console.error("Erro no /api/chat:", err);
    return res.status(500).json({ error: String(err) });
  }
}
