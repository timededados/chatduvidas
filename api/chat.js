// arquivo anterior que funcionava

// api/chat.js
import OpenAI from "openai";
import fs from "fs/promises";
import path from "path";
import { AsyncLocalStorage } from "async_hooks";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const DATA_DIR = path.join(process.cwd(), "data");
const BOOK_PATH = path.join(DATA_DIR, "abramede_texto.json");
const EMB_PATH = path.join(DATA_DIR, "abramede_embeddings.json");
const SUM_PATH = path.join(DATA_DIR, "sumario_final.json");

const EMB_MODEL = "text-embedding-3-small";
const CHAT_MODEL = "gpt-4o-mini";
const TOP_K = 6;
const MAX_CONTEXT_TOKENS = 3000;

// ==== Logging helpers (added) ====
const LOG_OPENAI = /^1|true|yes$/i.test(process.env.LOG_OPENAI || "");
const TRUNC_LIMIT = 800;
const als = new AsyncLocalStorage();

function truncate(str, n = TRUNC_LIMIT) {
  if (typeof str !== "string") return str;
  return str.length > n ? str.slice(0, n) + `... [${str.length - n} more chars]` : str;
}
function logSection(title) {
  const store = als.getStore();
  if (!(store && store.enabled)) return;
  if (store.logs) store.logs.push(`=== ${title} ===`);
  console.log(`\n=== ${title} ===`);
}
function logObj(label, obj) {
  const store = als.getStore();
  if (!(store && store.enabled)) return;
  let rendered;
  try { rendered = JSON.stringify(obj, null, 2); } catch { rendered = String(obj); }
  if (store.logs) store.logs.push(`${label}: ${rendered}`);
  console.log(label, rendered);
}
// Novo helper opcional para linhas simples
function logLine(...args) {
  const store = als.getStore();
  if (!(store && store.enabled)) return;
  const msg = args.map(a => {
    if (typeof a === "string") return a;
    try { return JSON.stringify(a); } catch { return String(a); }
  }).join(" ");
  store.logs.push(msg);
  console.log(msg);
}
function logOpenAIRequest(kind, payload) {
  const store = als.getStore();
  if (!(store && store.enabled)) return;
  const clone = { ...payload };
  if (Array.isArray(clone.messages)) {
    clone.messages = clone.messages.map(m => ({
      role: m.role,
      content: truncate(m.content, 600)
    }));
  }
  if (typeof clone.input === "string") clone.input = truncate(clone.input, 600);
  logSection(`Requisição OpenAI: ${kind}`);
  logObj("payload", clone);
}
function logOpenAIResponse(kind, resp, extra = {}) {
  const store = als.getStore();
  if (!(store && store.enabled)) return;
  const safe = {
    id: resp.id,
    model: resp.model,
    usage: resp.usage,
    created: resp.created,
    choices: (resp.choices || []).map(c => ({
      index: c.index,
      finish_reason: c.finish_reason,
      message: c.message ? {
        role: c.message.role,
        content: truncate(c.message.content, 600)
      } : undefined
    })),
    ...extra
  };
  logSection(`Resposta OpenAI: ${kind}`);
  logObj("data", safe);
}
// ==== End logging helpers ====

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

// Índice de sumário com acrônimos e sinônimos
function buildSummaryIndex(sumario) {
  const index = new Map();
  const addKey = (key, pages) => {
    const k = normalizeStr(key).trim();
    if (!k) return;
    const set = index.get(k) || new Set();
    (pages || []).forEach(p => set.add(p));
    index.set(k, set);
  };

  // Indexa tópicos e subcapítulos + acrônimos
  for (const top of sumario || []) {
    const topPages = top?.paginas || [];
    if (top?.topico) addKey(top.topico, topPages);

    for (const st of top?.subtopicos || []) {
      const stPages = (st?.paginas && st.paginas.length ? st.paginas : topPages);
      if (st?.titulo) {
        addKey(st.titulo, stPages);
        const tokens = normalizeStr(st.titulo).split(/\W+/).filter(Boolean);
        if (tokens.length >= 2) {
          const acronym = tokens.map(w => w[0]).join("");
          addKey(acronym, stPages); // ex: "Ressuscitação cardiopulmonar" -> "rcp"
        }
      }
    }
  }

  // Heurísticas leves: sinônimos comuns mapeados às páginas corretas via sumário
  const findPagesBySubtopicTitle = (needleNorm) => {
    for (const top of sumario || []) {
      for (const st of top?.subtopicos || []) {
        if (normalizeStr(st?.titulo || "") === needleNorm) {
          return st?.paginas || [];
        }
      }
    }
    return [];
  };

  // RCP
  const rcpPages = findPagesBySubtopicTitle("ressuscitacao cardiopulmonar");
  if (rcpPages.length) {
    [
      "massagem cardiaca",
      "compressao toracica",
      "compressoes toracicas",
      "cpr",
      "parada cardiorrespiratoria",
      "ressuscitacao cardiopulmonar",
      "rcp"
    ].forEach(k => addKey(k, rcpPages));
  }

  return index;
}

// Busca no sumário (agora usando índice com acrônimos/sinônimos)
function searchSummary(sumario, query) {
  const idx = buildSummaryIndex(sumario);
  const nq = normalizeStr(query);
  const hits = new Set();
  for (const [key, pages] of idx.entries()) {
    if (key && nq.includes(key)) {
      pages.forEach(p => hits.add(p));
    }
  }
  return Array.from(hits).sort((a, b) => a - b);
}

// ---------- Função principal ----------
export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();

  // Sempre habilitar logs para expor toda a interação
  // const forceDebug = /^(1|true|yes|on)$/i.test(String(req.query?.debug ?? req.body?.debug ?? ""));
  als.enterWith({ logs: [], enabled: true });
  const getLogs = () => (als.getStore()?.logs || []);

  try {
    const { question } = req.body;
    if (!question || !question.trim())
      return res.status(400).json({ error: "Pergunta vazia", logs: getLogs() });

    const storeEnabled = als.getStore()?.enabled;
    if (storeEnabled) {
      logSection("Pergunta recebida");
      logObj("question", question);
    }

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

    // 3️⃣ Busca no sumário (reforçada com acrônimos/sinônimos)
    const pagesFromSummary = searchSummary(sumario, question);
    if (als.getStore()?.enabled) {
      logSection("Páginas do sumário");
      logObj("pagesFromSummary", pagesFromSummary);
    }

    // 4️⃣ Consulta de embedding única
    const embReq = { model: EMB_MODEL, input: question };
    logOpenAIRequest("embeddings.create", embReq);
    const tEmb0 = Date.now();
    const qEmbResp = await openai.embeddings.create(embReq);
    const embMs = Date.now() - tEmb0;
    logOpenAIResponse("embeddings.create", qEmbResp, {
      duration_ms: embMs,
      embedding_dim: qEmbResp.data?.[0]?.embedding?.length
    });
    const queryEmb = qEmbResp.data[0].embedding;

    const qNorm = normalizeStr(question);
    const qTokens = Array.from(
      new Set(qNorm.split(/\W+/).filter(t => t && t.length > 2))
    );

    // 4.1️⃣ Define conjunto de candidatos:
    // - Se achou páginas no sumário, restringe a elas e vizinhas (±2) para evitar desvio para seções distantes.
    // - Caso contrário, considera todas as páginas.
    const embByPage = new Map(pageEmbeddings.map(pe => [pe.pagina, pe.embedding]));
    let candidatePages;
    if (pagesFromSummary.length) {
      const s = new Set();
      for (const p of pagesFromSummary) {
        s.add(p);
        s.add(p - 2); s.add(p - 1); s.add(p + 1); s.add(p + 2);
      }
      candidatePages = Array.from(s).filter(p => pageMap.has(p) && embByPage.has(p));
    } else {
      candidatePages = pageEmbeddings.map(pe => pe.pagina).filter(p => pageMap.has(p));
    }
    if (als.getStore()?.enabled) {
      logSection("Candidatos (embedding)");
      logObj("candidatePages_count", candidatePages.length);
    }

    // Calcular scores por página (apenas nos candidatos)
    let minEmb = Infinity, maxEmb = -Infinity, maxLex = 0;
    const prelim = [];
    for (const pg of candidatePages) {
      const peEmb = embByPage.get(pg);
      if (!peEmb) continue;
      const embScore = cosineSim(queryEmb, peEmb);
      const raw = pageMap.get(pg) || "";
      const txt = normalizeStr(raw);
      let lexScore = 0;
      for (const t of qTokens) lexScore += countOccurrences(txt, t);
      prelim.push({
        pagina: pg,
        embScore,
        lexScore,
        inSummary: pagesFromSummary.includes(pg)
      });
      if (embScore < minEmb) minEmb = embScore;
      if (embScore > maxEmb) maxEmb = embScore;
      if (lexScore > maxLex) maxLex = lexScore;
    }

    const ranked = prelim.map(r => {
      const embNorm = (r.embScore - minEmb) / (Math.max(1e-8, maxEmb - minEmb));
      const lexNorm = maxLex > 0 ? r.lexScore / maxLex : 0;

      // Se temos páginas do sumário, aumentamos fortemente o peso delas
      const summaryBoost = r.inSummary ? (pagesFromSummary.length ? 0.5 : 0.08) : 0;

      const finalScore = 0.7 * embNorm + 0.3 * lexNorm + summaryBoost;
      return { ...r, embNorm, lexNorm, finalScore };
    }).sort((a, b) => {
      if (b.finalScore !== a.finalScore) return b.finalScore - a.finalScore;
      return a.pagina - b.pagina; // desempate determinístico
    });

    if (!ranked.length) {
      return res.json({ answer: "Não encontrei conteúdo no livro.", used_pages: [], logs: getLogs() });
    }

    if (als.getStore()?.enabled && ranked.length) {
      const top = ranked[0];
      logSection("Ranqueamento - Top 1");
      logObj("pagina_topo", {
        pagina: top.pagina,
        embScore: top.embScore,
        lexScore: top.lexScore,
        embNorm: top.embNorm,
        lexNorm: top.lexNorm,
        finalScore: top.finalScore,
        inSummary: top.inSummary
      });
    }

    // 5️⃣ Seleciona até 2 páginas para contexto
    const selectedPages = ranked.slice(0, Math.min(2, ranked.length)).map(r => r.pagina);
    const nonEmptyPages = selectedPages.filter(p => (pageMap.get(p) || "").trim());
    if (!nonEmptyPages.length) {
      return res.json({ answer: "Não encontrei conteúdo no livro.", used_pages: [], logs: getLogs() });
    }
    if (als.getStore()?.enabled) {
      logSection("Páginas selecionadas para contexto");
      logObj("selectedPages", nonEmptyPages);
    }

    // 6️⃣ Monta o contexto (1 ou 2 páginas)
    const contextText = nonEmptyPages.map(p =>
      `--- Página ${p} ---\n${(pageMap.get(p) || "").trim()}\n`
    ).join("\n");
    if (als.getStore()?.enabled) {
      logSection("Contexto bruto");
      logObj("contextText_trunc", truncate(contextText, 1000));
    }

    // 7️⃣ Prompt restritivo multi-página
    const systemInstruction = `
Você responde exclusivamente com base nos textos abaixo (até 2 páginas).
Regras:
- Não adicione informações externas.
- Use somente frases originais dos textos fornecidos.
- Avalie cada página separadamente.
- Se somente uma página contiver a resposta, cite apenas "Página X" e use pelo menos 1 trecho literal entre aspas dessa página.
- Se as duas páginas tiverem partes relevantes, combine a resposta citando claramente ambas (ex: Página 10: "..." / Página 11: "...").
- Use sempre "Página X" ao citar.
- Não invente página que não está no contexto.
- Se nenhuma página contiver a resposta, responda exatamente: "Não encontrei conteúdo no livro."
`.trim();

    const userPrompt = `
Conteúdo do livro (1 ou 2 páginas):
${contextText}

Pergunta do usuário:
"""${question}"""

Instruções de resposta:
1. Indique apenas as páginas que realmente suportam a resposta.
2. Use somente trechos literais entre aspas exatamente como aparecem.
3. Se as duas páginas forem úteis, una-as citando ambas separadamente.
4. Se só uma tiver informação útil, cite apenas essa.
5. Caso nenhuma tenha a resposta: "Não encontrei conteúdo no livro."
`.trim();

    // 8️⃣ Geração determinística
    const chatReq = {
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
      seed: seedFromString(question)
    };
    logOpenAIRequest("chat.completions.create", chatReq);
    const tChat0 = Date.now();
    const chatResp = await openai.chat.completions.create(chatReq);
    const chatMs = Date.now() - tChat0;
    logOpenAIResponse("chat.completions.create", chatResp, { duration_ms: chatMs });

    const answer =
      chatResp.choices?.[0]?.message?.content?.trim() ||
      "Não encontrei conteúdo no livro.";

    if (als.getStore()?.enabled) {
      logSection("Resposta final");
      logObj("payload", { answer, used_pages: nonEmptyPages });
    }

    return res.status(200).json({
      answer,
      used_pages: nonEmptyPages,
      logs: getLogs()
    });

  } catch (err) {
    console.error("Erro no /api/chat:", err);
    return res.status(500).json({ error: String(err), logs: getLogs() });
  }
}
