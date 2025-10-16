// api/chat.js
import OpenAI from "openai";
import fs from "fs/promises";
import path from "path";
import { AsyncLocalStorage } from "async_hooks";
import { createClient } from "@supabase/supabase-js"; // novo

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const DATA_DIR = path.join(process.cwd(), "data");
const BOOK_PATH = path.join(DATA_DIR, "abramede_texto.json");
const EMB_PATH = path.join(DATA_DIR, "abramede_embeddings.json");
const SUM_PATH = path.join(DATA_DIR, "sumario_final.json");
// Novo: fallback local do dicionário (quando não há Supabase)
const DICT_PATH = path.join(DATA_DIR, "dictionary.json");

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

// ==== Dicionário: carregar, ranquear e formatar (novo) ====
// Supabase (se disponível)
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey =
  process.env.SUPABASE_SERVICE_ROLE_KEY ||
  process.env.SUPABASE_SERVICE_ROLE ||
  process.env.SUPABASE_ANON_KEY;
const supabase =
  supabaseUrl && supabaseKey ? createClient(supabaseUrl, supabaseKey) : null;

async function loadDictionaryFromSupabase() {
  if (!supabase) return null;
  try {
    const { data, error } = await supabase
      .from("dictionary")
      .select("*")
      .order("created_at", { ascending: false });
    if (error) throw error;
    const items = (data || []).map(item => ({
      id: item.id,
      titulo: item.titulo,
      autor: item.autor,
      tipoConteudo: item.tipo_conteudo,
      pago: item.pago,
      link: item.link,
      tags: item.tags || [],
      imagemUrl: item.imagem_url || null,
      createdAt: item.created_at,
      updatedAt: item.updated_at
    }));
    return items;
  } catch (e) {
    logLine("Supabase dictionary load error:", String(e?.message || e));
    return null;
  }
}

async function loadDictionaryFromFile() {
  try {
    const raw = await fs.readFile(DICT_PATH, "utf8");
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return [];
    // normaliza chaves do arquivo local
    return arr.map(x => ({
      id: x.id,
      titulo: x.titulo,
      autor: x.autor,
      tipoConteudo: x.tipoConteudo || x.tipo_conteudo,
      pago: !!x.pago,
      link: x.link,
      tags: Array.isArray(x.tags) ? x.tags : [],
      imagemUrl: x.imagemUrl || x.imagem_url || null,
      createdAt: x.createdAt || x.created_at,
      updatedAt: x.updatedAt || x.updated_at
    }));
  } catch {
    return [];
  }
}

async function loadDictionaryEntries() {
  // tenta Supabase primeiro; se falhar, usa arquivo local
  const sb = await loadDictionaryFromSupabase();
  if (sb && Array.isArray(sb)) return sb;
  const file = await loadDictionaryFromFile();
  return file;
}

function rankDictItems(question, items, maxItems = 3) {
  const qNorm = normalizeStr(question);
  const qTokens = Array.from(new Set(qNorm.split(/\W+/).filter(t => t && t.length > 2)));
  const scored = [];

  for (const it of items || []) {
    const title = normalizeStr(it.titulo || "");
    const author = normalizeStr(it.autor || "");
    const tipo = normalizeStr(it.tipoConteudo || "");
    const tagsNorm = (it.tags || []).map(t => normalizeStr(String(t || "")));
    const blob = [title, author, tipo, tagsNorm.join(" ")].join(" ").trim();

    let score = 0;

    // peso por tokens no título/tipo/autor
    for (const t of qTokens) {
      if (countOccurrences(title, t)) score += 2; // título mais relevante
      if (countOccurrences(tipo, t)) score += 1.2;
      if (countOccurrences(author, t)) score += 0.8;
      if (countOccurrences(blob, t)) score += 0.3;
    }

    // peso por tags (tags são categorias do sumário)
    for (const tg of tagsNorm) {
      if (!tg) continue;
      if (qNorm.includes(tg)) score += 3.0;
      // correspondência por token
      for (const t of qTokens) {
        if (tg === t) score += 1.8;
      }
    }

    // pequeno bônus se tiver link (útil para o usuário)
    if (it.link) score += 0.2;

    if (score > 0) {
      scored.push({ item: it, score });
    }
  }

  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, maxItems);
}

function formatDictItemsForPrompt(scored) {
  if (!scored?.length) return "";
  const lines = scored.map(({ item }) => {
    const tagStr = (item.tags || []).join(", ");
    const pagoStr = item.pago ? "Sim" : "Não";
    const linkStr = item.link ? item.link : "—";
    return `- Título: ${item.titulo} | Tipo: ${item.tipoConteudo || "—"} | Pago: ${pagoStr} | Autor: ${item.autor || "—"} | Tags: ${tagStr || "—"} | Link: ${linkStr}`;
  });
  return lines.join("\n");
}
// ==== fim bloco dicionário ====

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

    // ==== NOVO: carregar e ranquear itens do dicionário pela pergunta ====
    const dictAll = await loadDictionaryEntries();
    if (als.getStore()?.enabled) {
      logSection("Dicionário - total de itens carregados");
      logObj("count", dictAll.length);
    }
    const dictTop = rankDictItems(question, dictAll, 3);
    const dictUsed = dictTop.map(d => ({ id: d.item.id, score: Number(d.score.toFixed(3)) }));
    if (als.getStore()?.enabled) {
      logSection("Dicionário - itens relevantes");
      logObj("dictUsed", dictUsed);
    }
    const dictContextText = formatDictItemsForPrompt(dictTop);
    // ================================================================

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
      // Sem páginas candidatas; se houver sugestões do dicionário, retorne-as
      if (dictTop.length) {
        const suggestions = dictContextText ? `\n\nSugestões do dicionário:\n${dictContextText}` : "";
        return res.json({
          answer: "Não encontrei conteúdo no livro." + suggestions,
          used_pages: [],
          used_dict: dictUsed,
          logs: getLogs()
        });
      }
      return res.json({ answer: "Não encontrei conteúdo no livro.", used_pages: [], used_dict: [], logs: getLogs() });
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
      // Sem conteúdo de livro; mas temos dicionário?
      if (dictTop.length) {
        const suggestions = dictContextText ? `\n\nSugestões do dicionário:\n${dictContextText}` : "";
        return res.json({
          answer: "Não encontrei conteúdo no livro." + suggestions,
          used_pages: [],
          used_dict: dictUsed,
          logs: getLogs()
        });
      }
      return res.json({ answer: "Não encontrei conteúdo no livro.", used_pages: [], used_dict: [], logs: getLogs() });
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

    // 7️⃣ Prompt restritivo multi-página (ajustado para também incluir dicionário)
    const systemInstruction = `
Você responde exclusivamente com base nos conteúdos fornecidos abaixo.
Fontes:
1) Livro (1–2 páginas): use somente trechos literais entre aspas e cite sempre "Página X".
2) Dicionário (itens opcionais): liste sugestões no final se houver, usando apenas as informações fornecidas (título, autor, tipo, pago, tags e link). Não confunda com o livro e não invente nada.

Regras:
- Não adicione informações externas.
- Avalie cada página do livro separadamente.
- Se somente uma página contiver a resposta, cite apenas "Página X" e inclua pelo menos 1 trecho literal entre aspas dessa página.
- Se as duas páginas tiverem partes relevantes, combine citando claramente ambas (ex: Página 10: "..." / Página 11: "...").
- Se nenhuma página do livro contiver a resposta, diga exatamente: "Não encontrei conteúdo no livro." e, se houver, inclua em seguida as "Sugestões do dicionário".
`.trim();

    const userPrompt = `
Conteúdo do livro (1 ou 2 páginas):
${contextText}

Itens do dicionário relevantes (se houver; para sugestões no final):
${dictContextText || "(nenhum)"}

Pergunta do usuário:
"""${question}"""

Como responder:
1. Responda apenas com base no CONTEÚDO DO LIVRO acima, citando "Página X" e trechos literais entre aspas.
2. Após a resposta do livro, se houver "Itens do dicionário", inclua uma seção final intitulada "Sugestões do dicionário" listando até ${dictTop.length || 0} itens com título e link, e opcionalmente tipo/pago/tags. Não misture as fontes.
3. Se o livro não responder à pergunta, escreva: "Não encontrei conteúdo no livro." e então mostre "Sugestões do dicionário" (se houver).
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
      logObj("payload", { answer, used_pages: nonEmptyPages, used_dict: dictUsed });
    }

    return res.status(200).json({
      answer,
      used_pages: nonEmptyPages,
      used_dict: dictUsed,
      logs: getLogs()
    });

  } catch (err) {
    console.error("Erro no /api/chat:", err);
    return res.status(500).json({ error: String(err), logs: getLogs() });
  }
}
