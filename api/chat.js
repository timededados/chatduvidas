// api/chat.js
import OpenAI from "openai";
import fs from "fs/promises";
import path from "path";
import { AsyncLocalStorage } from "async_hooks";
import { toFile } from "openai/uploads";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const DATA_DIR = path.join(process.cwd(), "data");
const BOOK_PATH = path.join(DATA_DIR, "abramede_texto.json");
const EMB_PATH = path.join(DATA_DIR, "abramede_embeddings.json");
const SUM_PATH = path.join(DATA_DIR, "sumario_final.json");

const EMB_MODEL = "text-embedding-3-small";
const CHAT_MODEL = "gpt-4o-mini";
const TOP_K = 6;
const MAX_CONTEXT_TOKENS = 3000;

// +++ Novo: limites para recomendação do dicionário +++
const DICT_MAX_CANDIDATES = 20;   // candidatos enviados ao modelo
const DICT_MAX_RECOMMEND = 5;     // máximo de recomendações finais

// +++ NOVO: Configuração de expansão de contexto +++
const EXPAND_CONTEXT = true;      // Ativar expansão de páginas adjacentes
const ADJACENT_RANGE = 1;         // Quantas páginas antes/depois incluir (1 = uma antes e uma depois)

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

// Novo: extrai páginas citadas no texto final (ex.: "Página 10", "(p. 10)")
function extractCitedPages(text) {
  if (!text) return [];
  const set = new Set();
  const patterns = [
    /página\s+(\d+)/gi,    // "Página 123"
    /pagina\s+(\d+)/gi,    // "Pagina 123" (sem acento)
    /\(p\.\s*(\d+)\)/gi    // "(p. 123)"
  ];
  for (const re of patterns) {
    let m;
    while ((m = re.exec(text)) !== null) {
      const n = parseInt(m[1], 10);
      if (!isNaN(n)) set.add(n);
    }
  }
  return Array.from(set).sort((a, b) => a - b);
}

// +++ NOVA FUNÇÃO: Expande páginas com adjacentes +++
function expandWithAdjacentPages(selectedPages, pageMap, range = ADJACENT_RANGE) {
  const expandedSet = new Set();
  
  for (const page of selectedPages) {
    // Adiciona a página original
    expandedSet.add(page);
    
    // Adiciona páginas anteriores
    for (let i = 1; i <= range; i++) {
      const prevPage = page - i;
      if (pageMap.has(prevPage)) {
        expandedSet.add(prevPage);
      }
    }
    
    // Adiciona páginas posteriores
    for (let i = 1; i <= range; i++) {
      const nextPage = page + i;
      if (pageMap.has(nextPage)) {
        expandedSet.add(nextPage);
      }
    }
  }
  
  // Retorna array ordenado
  return Array.from(expandedSet).sort((a, b) => a - b);
}

// Adicionado: funções de busca no sumário (faltavam)
function buildSummaryIndex(sumario) {
  const index = new Map();
  const addKey = (key, pages) => {
    const k = normalizeStr(key).trim();
    if (!k) return;
    const set = index.get(k) || new Set();
    (pages || []).forEach(p => set.add(p));
    index.set(k, set);
  };

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
          addKey(acronym, stPages);
        }
      }
    }
  }

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

// +++ Novo: helpers para recomendação do dicionário +++
function buildBaseUrl(req) {
  const proto = req.headers["x-forwarded-proto"] || "http";
  const host = req.headers.host || "localhost";
  return `${proto}://${host}`;
}

function scoreDictItem(item, qTokens) {
  const parts = [
    item.titulo || "",
    item.autor || "",
    item.tipoConteudo || item.tipo_conteudo || "",
    Array.isArray(item.tags) ? item.tags.join(" ") : ""
  ];
  const text = normalizeStr(parts.join(" | "));
  let score = 0;
  for (const t of qTokens) {
    // prioriza match de tokens do título e tags
    const inTitulo = countOccurrences(normalizeStr(item.titulo || ""), t);
    const inTags = countOccurrences(normalizeStr((item.tags || []).join(" ")), t);
    const inRest = countOccurrences(text, t);
    score += inTitulo * 3 + inTags * 2 + Math.max(inRest, 0);
  }
  return score;
}

function pickTopDictCandidates(items, question, limit = DICT_MAX_CANDIDATES) {
  const qNorm = normalizeStr(question);
  const qTokens = Array.from(new Set(qNorm.split(/\W+/).filter(w => w && w.length > 2)));
  const withScores = (items || []).map(it => ({ it, s: scoreDictItem(it, qTokens) }));
  withScores.sort((a, b) => b.s - a.s);
  return withScores.slice(0, limit).map(x => x.it);
}

// Adicionado: helpers para escapar HTML/atributos
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;" }[c]));
}
function escapeAttr(s) {
  return String(s).replace(/"/g, "&quot;");
}

// +++ Novo: mapeia tipo de conteúdo -> rótulo e estilo do botão
function buttonForType(tipoRaw, isPremium) {
  const tipo = String(tipoRaw || "").toLowerCase();
  if (tipo.includes("podteme")) return { label: "🎧 Ouvir episódio", kind: "primary" };
  if (tipo.includes("preparatório teme") || tipo.includes("preparatorio teme")) return { label: "▶️ Assistir aula", kind: "accent" };
  if (tipo.includes("instagram")) return { label: "📱 Ver post", kind: "primary" };
  if (tipo.includes("blog")) return { label: "📰 Ler artigo", kind: "primary" };
  if (tipo.includes("curso")) return { label: isPremium ? "💎 Conhecer o curso" : "▶️ Acessar curso", kind: isPremium ? "premium" : "accent" };
  return { label: "🔗 Acessar conteúdo", kind: isPremium ? "premium" : "primary" };
}

// +++ LAYOUT CORRIGIDO: HTML compacto e limpo
function btnStyle(kind) {
  const base = "display:inline-block;padding:8px 12px;border-radius:8px;text-decoration:none;font-weight:500;font-size:13px;border:1px solid;cursor:pointer;";
  if (kind === "accent") return base + "background:rgba(56,189,248,0.08);border-color:rgba(56,189,248,0.25);color:#38bdf8;";
  if (kind === "premium") return base + "background:rgba(245,158,11,0.08);border-color:rgba(245,158,11,0.25);color:#f59e0b;";
  return base + "background:rgba(34,197,94,0.08);border-color:rgba(34,197,94,0.25);color:#22c55e;";
}

// +++ LAYOUT CORRIGIDO: renderiza lista de itens com HTML mínimo
function renderDictItemsList(items, isPremiumSection) {
  if (!items.length) return "";
  
  const itemsHtml = items.map(it => {
    const titulo = escapeHtml(it.titulo || "");
    const autor = it.autor ? ` <span style="color:#94a3b8">— ${escapeHtml(it.autor)}</span>` : "";
    const tipo = it.tipoConteudo || it.tipo_conteudo || "";
    const { label, kind } = buttonForType(tipo, !!it.pago);
    const href = it.link ? ` href="${escapeAttr(it.link)}" target="_blank"` : "";
    const btn = it.link ? `<div style="margin-top:6px"><a style="${btnStyle(kind)}"${href}>${label}</a></div>` : "";
    
    // Adiciona badges para conteúdo premium
    const badges = isPremiumSection ? 
      `<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px"><span style="border:1px dashed #1f2937;border-radius:999px;padding:4px 8px;font-size:11px;color:#94a3b8">Carga horária: 12h</span><span style="border:1px dashed #1f2937;border-radius:999px;padding:4px 8px;font-size:11px;color:#94a3b8">Aulas on-demand</span><span style="border:1px dashed #1f2937;border-radius:999px;padding:4px 8px;font-size:11px;color:#94a3b8">Certificado</span></div>` : "";
    
    return `<div style="padding:10px;border:1px solid #1f2937;border-radius:8px;background:rgba(255,255,255,0.015);margin-bottom:8px"><div><strong>${titulo}</strong>${autor}</div>${btn}${badges}</div>`;
  }).join("");
  
  const color = isPremiumSection ? "#f59e0b" : "#22c55e";
  const label = isPremiumSection ? "Conteúdo premium (opcional)" : "Conteúdo complementar";
  
  return `<section style="background:linear-gradient(180deg,#0b1220,#111827);border:1px solid #1f2937;border-radius:12px;padding:14px;margin-bottom:12px"><span style="display:inline-flex;align-items:center;gap:6px;padding:5px 9px;border-radius:999px;border:1px solid #1f2937;background:rgba(255,255,255,0.02);color:#cbd5e1;font-weight:600;font-size:11px;letter-spacing:0.3px;text-transform:uppercase"><span style="width:6px;height:6px;border-radius:50%;background:${color}"></span>${label}</span><div style="margin-top:10px">${itemsHtml}</div></section>`;
}

// +++ LAYOUT CORRIGIDO: HTML final ultra compacto
function renderFinalHtml({ bookAnswer, citedPages, dictItems }) {
  // Header conciso com cores ajustadas para fundo verde
  const header = `<header style="margin-bottom:14px"><h1 style="font-size:18px;margin:0 0 6px 0;font-weight:600;color:#1a1a1a">Encontrei a informação que responde à sua dúvida 👇</h1></header>`;

  // Livro - seção principal
  const bookSection = `<section style="background:linear-gradient(180deg,#0b1220,#111827);border:1px solid #1f2937;border-radius:12px;padding:14px;margin-bottom:12px"><span style="display:inline-flex;align-items:center;gap:6px;padding:5px 9px;border-radius:999px;border:1px solid #1f2937;background:rgba(255,255,255,0.02);color:#cbd5e1;font-weight:600;font-size:11px;letter-spacing:0.3px;text-transform:uppercase"><span style="width:6px;height:6px;border-radius:50%;background:#38bdf8"></span>Livro (fonte principal)</span><div style="position:relative;padding:12px 14px;border-left:3px solid #38bdf8;background:rgba(56,189,248,0.06);border-radius:6px;line-height:1.5;margin-top:10px"><div>${escapeHtml(bookAnswer).replace(/\n/g, "<br>")}</div><small style="display:block;color:#94a3b8;margin-top:6px;font-size:11px">Trechos do livro-base do curso.</small></div></section>`;

  // Separar e renderizar itens
  const freeItems = (dictItems || []).filter(x => !x.pago);
  const premiumItems = (dictItems || []).filter(x => x.pago);
  
  let content = header + bookSection;
  if (freeItems.length) content += renderDictItemsList(freeItems, false);
  if (premiumItems.length) content += renderDictItemsList(premiumItems, true);

  return `<div style="max-width:680px;font-family:system-ui,-apple-system,sans-serif;color:#e5e7eb">${content}</div>`;
}

// Adicionado: recomendação a partir do dicionário (retorna apenas os itens selecionados)
async function recommendFromDictionary(req, question) {
  try {
    const baseUrl = buildBaseUrl(req);
    const res = await fetch(`${baseUrl}/api/dict`);
    if (!res.ok) throw new Error(`GET /api/dict falhou: ${res.status}`);
    const dictItems = await res.json();
    if (!Array.isArray(dictItems) || dictItems.length === 0) return { raw: [] };

    logSection("Dicionário - total carregado");
    logObj("count", dictItems.length);

    // pré-filtro lexical
    const candidates = pickTopDictCandidates(dictItems, question, DICT_MAX_CANDIDATES);
    logSection("Dicionário - candidatos enviados ao modelo");
    logObj("candidates_count", candidates.length);

    // payload enxuto para o modelo
    const slim = candidates.map(it => ({
      id: it.id,
      titulo: it.titulo,
      autor: it.autor || "",
      tipo: it.tipoConteudo || it.tipo_conteudo || "",
      tags: Array.isArray(it.tags) ? it.tags : [],
      link: it.link || "",
      pago: !!it.pago
    }));

    const system = `
Você seleciona itens de um dicionário relevantes para a pergunta do usuário.
Critérios:
- Escolha no máximo ${DICT_MAX_RECOMMEND} itens bem relacionados ao tema da pergunta.
- Dê preferência a correspondências no título/tipo/tags.
- Se nada for claramente relevante, retorne lista vazia.
Responda EXCLUSIVAMENTE em JSON:
{"recommendedIds": ["id1","id2",...]}
`.trim();

    const user = `
Pergunta: """${question}"""

Itens (JSON):
${JSON.stringify(slim, null, 2)}
`.trim();

    const chatReq = {
      model: CHAT_MODEL,
      messages: [
        { role: "system", content: system },
        { role: "user", content: user }
      ],
      temperature: 0,
      top_p: 1,
      max_tokens: 200,
      seed: seedFromString(question + "|dict")
    };
    logOpenAIRequest("chat.completions.create [dict]", chatReq);
    const t0 = Date.now();
    const resp = await openai.chat.completions.create(chatReq);
    const ms = Date.now() - t0;
    logOpenAIResponse("chat.completions.create [dict]", resp, { duration_ms: ms });

    const raw = resp.choices?.[0]?.message?.content?.trim() || "{}";
    let ids = [];
    try {
      const m = raw.match(/\{[\s\S]*\}/);
      const parsed = JSON.parse(m ? m[0] : raw);
      if (Array.isArray(parsed.recommendedIds)) ids = parsed.recommendedIds.slice(0, DICT_MAX_RECOMMEND);
    } catch {
      ids = [];
    }

    const selected = ids
      .map(id => candidates.find(c => c.id === id))
      .filter(Boolean)
      .slice(0, DICT_MAX_RECOMMEND);

    const finalSel = selected.length ? selected : candidates.slice(0, Math.min(3, candidates.length));

    logSection("Dicionário - selecionados");
    logObj("ids", finalSel.map(x => x.id));

    return { raw: finalSel };
  } catch (e) {
    logSection("Dicionário - erro");
    logObj("error", String(e));
    return { raw: [] };
  }
}

// Para ambientes Next.js / Vercel: aumentar limite do body para áudio base64
export const config = {
  api: { bodyParser: { sizeLimit: "25mb" } }
};

// Adicionado: transcrição de áudio base64 com gpt-4o-mini-transcribe
async function transcribeBase64AudioToText(audioStr, mime = "audio/webm") {
  try {
    logSection("Transcrição de áudio");
    const clean = String(audioStr || "").replace(/^data:.*;base64,/, "");
    logObj("audio_base64_len", clean.length);
    const buf = Buffer.from(clean, "base64");
    logObj("audio_bytes", buf.length);
    const ext = mime.includes("mpeg") ? "mp3"
      : mime.includes("wav") ? "wav"
      : mime.includes("ogg") ? "ogg"
      : mime.includes("m4a") ? "m4a"
      : "webm";
    const filename = `audio.${ext}`;
    const file = await toFile(buf, filename, { type: mime });
    const t0 = Date.now();
    const resp = await openai.audio.transcriptions.create({
      model: "gpt-4o-mini-transcribe",
      file,
      language: "pt"
    });
    const ms = Date.now() - t0;
    logObj("transcription_ms", ms);
    const text = (resp && (resp.text || resp.transcript || resp?.results?.[0]?.transcript)) || "";
    logObj("transcription_preview", truncate(text, 200));
    return text.trim();
  } catch (e) {
    logSection("Transcrição de áudio - erro");
    logObj("error", String(e));
    return "";
  }
}

// ---------- Função principal ----------
export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();

  // Sempre habilitar logs para expor toda a interação
  // const forceDebug = /^(1|true|yes|on)$/i.test(String(req.query?.debug ?? req.body?.debug ?? ""));
  als.enterWith({ logs: [], enabled: true });
  const getLogs = () => (als.getStore()?.logs || []);

  try {
    // Novo: aceitar pergunta por voz (base64/data URL)
    const { question: questionRaw, audio, audio_mime } = req.body || {};
    let question = String(questionRaw || "").trim();
    if (!question && audio) {
      logSection("Entrada de áudio detectada");
      question = await transcribeBase64AudioToText(audio, audio_mime || "audio/webm");
    }
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
    const pagesFromSummary = (typeof searchSummary === "function")
      ? searchSummary(sumario, question)
      : [];
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

    // ====================== BOOST LEXICAL AJUSTADO ======================

    // fator de amplificação do peso lexical (1.0 = neutro, >1 aumenta peso das palavras da pergunta)
    const LEXICAL_BOOST = 5.0;

    // valor base do boost para páginas vindas do sumário
    const SUMMARY_BOOST_BASE = pagesFromSummary.length ? 0.5 : 0.08;

    const ranked = prelim
      .map(r => {
        const embNorm = (r.embScore - minEmb) / Math.max(1e-8, maxEmb - minEmb);
        const lexNorm = maxLex > 0 ? r.lexScore / maxLex : 0;

        // 🔹 camada de boost lexical: amplifica ligeiramente o impacto de lexNorm
        const lexicalBoostFactor = 1 + (lexNorm * (LEXICAL_BOOST - 1));

        const summaryBoost = r.inSummary ? SUMMARY_BOOST_BASE : 0;

        // 🔹 pontuação final combinando embeddings, lexNorm e boosts
        const finalScore = (0.7 * embNorm + 0.3 * lexNorm) * lexicalBoostFactor + summaryBoost;

        return { ...r, embNorm, lexNorm, finalScore, lexicalBoostFactor };
      })
      .sort((a, b) => {
        if (b.finalScore !== a.finalScore) return b.finalScore - a.finalScore;
        return a.pagina - b.pagina; // desempate determinístico
      });

    // =====================================================================

    if (!ranked.length) {
      return res.json({
        answer: "Não encontrei conteúdo no livro.",
        used_pages: [],
        question_used: question,
        logs: getLogs()
      });
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

    // 5️⃣ Seleciona até 2 páginas principais
    const selectedPages = ranked.slice(0, Math.min(2, ranked.length)).map(r => r.pagina);
    
    // +++ NOVO: Expande com páginas adjacentes +++
    let finalPages;
    if (EXPAND_CONTEXT) {
      finalPages = expandWithAdjacentPages(selectedPages, pageMap, ADJACENT_RANGE);
      
      if (als.getStore()?.enabled) {
        logSection("Expansão de contexto");
        logObj("original_pages", selectedPages);
        logObj("expanded_pages", finalPages);
        logObj("adjacent_range", ADJACENT_RANGE);
      }
    } else {
      finalPages = selectedPages;
    }
    
    // Filtra páginas não vazias
    const nonEmptyPages = finalPages.filter(p => (pageMap.get(p) || "").trim());
    
    if (!nonEmptyPages.length) {
      return res.json({
        answer: "Não encontrei conteúdo no livro.",
        used_pages: [],
        question_used: question,
        logs: getLogs()
      });
    }
    
    if (als.getStore()?.enabled) {
      logSection("Páginas finais para contexto");
      logObj("finalPages", nonEmptyPages);
      logObj("total_pages", nonEmptyPages.length);
    }

    // 6️⃣ Monta o contexto com todas as páginas (incluindo adjacentes)
    const contextText = nonEmptyPages.map(p =>
      `--- Página ${p} ---\n${(pageMap.get(p) || "").trim()}\n`
    ).join("\n");
    
    if (als.getStore()?.enabled) {
      logSection("Contexto bruto");
      logObj("contextText_length", contextText.length);
      logObj("contextText_trunc", truncate(contextText, 1000));
    }

    // 7️⃣ Prompt atualizado para lidar com múltiplas páginas
    const systemInstruction = `
Você é um assistente que responde exclusivamente com trechos literais de um livro-base.

Regras obrigatórias:
- NÃO explique, NÃO resuma, NÃO interprete, NÃO altere palavras.
- Responda SOMENTE com as citações literais extraídas do livro fornecido.
- Inclua cada trecho exatamente como está no texto original.
- Identifique cada trecho com o número da página (ex: "- Página 694: \"trecho...\"").
- NÃO adicione frases introdutórias, comentários ou resumos.
- Se houver mais de um trecho relevante, liste-os em ordem crescente de página.
- Se não houver trechos claramente relevantes, responda apenas "Nenhum trecho encontrado no livro.".

Formato final da resposta:
- Página N: "trecho literal 1"
- Página M: "trecho literal 2"
Trechos do livro-base do curso.
`.trim();

    const userPrompt = `
Pergunta: """${question}"""

Trechos disponíveis do livro (cada um contém número da página):
${finalPages.map(p => `Página ${p}:\n${pageMap.get(p)}`).join("\n\n")}

Com base APENAS nos trechos acima, recorte os trechos exatos que respondem diretamente à pergunta.
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
      logObj("payload", { 
        answer, 
        used_pages: nonEmptyPages,
        original_selection: selectedPages,
        expanded_context: EXPAND_CONTEXT
      });
    }

    // +++ Novo: etapa de recomendação do dicionário e concatenação da resposta +++
    const dictRec = await recommendFromDictionary(req, question);

    // Ajuste: detectar páginas realmente citadas na resposta para montar o template (se necessário futuramente)
    const notFound = answer === "Não encontrei conteúdo no livro.";
    const citedPages = extractCitedPages(answer);

    // Novo: renderização no template (ou similar)
    const finalAnswer = notFound
      ? answer
      : renderFinalHtml({ bookAnswer: answer, citedPages, dictItems: dictRec.raw });

    return res.status(200).json({
      answer: finalAnswer,
      used_pages: nonEmptyPages,
      original_pages: selectedPages,
      expanded_context: EXPAND_CONTEXT,
      question_used: question,
      logs: getLogs()
    });

  } catch (err) {
    console.error("Erro no /api/chat:", err);
    return res.status(500).json({ error: String(err), logs: getLogs() });
  }
}