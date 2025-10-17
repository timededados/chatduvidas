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

// Limites para recomendação do dicionário
const DICT_MAX_CANDIDATES = 20;
const DICT_MAX_RECOMMEND = 5;

// Configuração de expansão de contexto
const EXPAND_CONTEXT = true;
const ADJACENT_RANGE = 2;  // ✅ Aumentado de 1 para 2 (±2 páginas)
const TOP_PAGES_TO_SELECT = 4;  // ✅ Novo: selecionar TOP 4 em vez de TOP 2

// ==== Sistema de Logging ====
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
  try { 
    rendered = JSON.stringify(obj, null, 2); 
  } catch { 
    rendered = String(obj); 
  }
  if (store.logs) store.logs.push(`${label}: ${rendered}`);
  console.log(label, rendered);
}

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

// ==== Funções Auxiliares de Processamento ====
function dot(a, b) {
  return a.reduce((s, v, i) => s + v * b[i], 0);
}

function norm(a) {
  return Math.sqrt(a.reduce((s, x) => s + x * x, 0));
}

function cosineSim(a, b) {
  return dot(a, b) / (norm(a) * norm(b) + 1e-8);
}

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

function extractCitedPages(text) {
  if (!text) return [];
  const set = new Set();
  const patterns = [
    /página\s+(\d+)/gi,
    /pagina\s+(\d+)/gi,
    /\(p\.\s*(\d+)\)/gi
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

// ==== Expansão de Páginas Adjacentes ====
function expandWithAdjacentPages(selectedPages, pageMap, range = ADJACENT_RANGE) {
  const expandedSet = new Set();
  
  for (const page of selectedPages) {
    expandedSet.add(page);
    
    for (let i = 1; i <= range; i++) {
      const prevPage = page - i;
      if (pageMap.has(prevPage)) {
        expandedSet.add(prevPage);
      }
    }
    
    for (let i = 1; i <= range; i++) {
      const nextPage = page + i;
      if (pageMap.has(nextPage)) {
        expandedSet.add(nextPage);
      }
    }
  }
  
  return Array.from(expandedSet).sort((a, b) => a - b);
}

// ==== NOVA FUNÇÃO: Busca Semântica no Sumário com OpenAI ====
async function semanticSearchSummary(sumario, question) {
  try {
    logSection("Busca Semântica no Sumário - Início");
    
    // 1. Preparar estrutura simplificada do sumário para o modelo
    const summaryStructure = sumario.map(secao => {
      const categorias = (secao.categorias || []).map(cat => {
        const topicos = (cat.topicos || []).map(top => {
          const subtopicos = (top.subtopicos || []).map(sub => ({
            titulo: sub.titulo,
            paginas: sub.paginas || []
          }));
          
          return {
            topico: top.topico,
            paginas: top.paginas || [],
            subtopicos
          };
        });
        
        return {
          categoria: cat.categoria,
          paginas: cat.paginas || [],
          topicos
        };
      });
      
      return {
        secao: secao.secao,
        categorias
      };
    });

    // 2. Criar prompt para o modelo identificar seções relevantes
    const systemPrompt = `Você é um especialista em medicina de emergência e terapia intensiva.

Sua tarefa é analisar uma pergunta do usuário e identificar quais seções, categorias, tópicos e subtópicos de um sumário de livro são relevantes para responder essa pergunta.

IMPORTANTE:
- Considere sinônimos médicos (ex: PCR = parada cardiorrespiratória = parada cardíaca)
- Considere abreviações comuns (ex: IAM, AVC, TEP, etc)
- Pense em contexto clínico amplo (ex: "dor no peito" pode relacionar-se com IAM, dissecção de aorta, embolia pulmonar)
- Seja inclusivo: se houver dúvida se um tópico é relevante, inclua-o
- SEMPRE considere que a pergunta é sobre adultos, a menos que especifique pediatria

Responda EXCLUSIVAMENTE em JSON seguindo este formato:
{
  "relevant_paths": [
    {
      "secao": "nome da seção",
      "categoria": "nome da categoria",
      "topico": "nome do tópico (ou null se toda categoria é relevante)",
      "subtopico": "nome do subtópico (ou null se todo tópico é relevante)",
      "reasoning": "breve explicação de por que é relevante"
    }
  ]
}

Se nenhuma seção for claramente relevante, retorne: {"relevant_paths": []}`;

    const userPrompt = `Pergunta do usuário: """${question}"""

Estrutura do sumário:
${JSON.stringify(summaryStructure, null, 2)}

Identifique quais partes do sumário são relevantes para responder a pergunta.`;

    const chatReq = {
      model: CHAT_MODEL,
      messages: [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt }
      ],
      temperature: 0,
      top_p: 1,
      max_tokens: 2000,
      seed: seedFromString(question + "|summary")
    };

    logOpenAIRequest("chat.completions.create [semantic_summary]", chatReq);
    const t0 = Date.now();
    const resp = await openai.chat.completions.create(chatReq);
    const ms = Date.now() - t0;
    logOpenAIResponse("chat.completions.create [semantic_summary]", resp, { duration_ms: ms });

    // 3. Parsear resposta e extrair páginas
    const raw = resp.choices?.[0]?.message?.content?.trim() || "{}";
    let relevantPaths = [];
    
    try {
      const m = raw.match(/\{[\s\S]*\}/);
      const parsed = JSON.parse(m ? m[0] : raw);
      relevantPaths = parsed.relevant_paths || [];
    } catch (e) {
      logSection("Busca Semântica no Sumário - Erro ao parsear JSON");
      logObj("error", String(e));
      logObj("raw_response", raw);
      return { pages: [], paths: [] };
    }

    logSection("Busca Semântica no Sumário - Caminhos Identificados");
    logObj("relevant_paths", relevantPaths);

    // 4. Coletar todas as páginas dos caminhos identificados
    const pagesSet = new Set();
    
    for (const path of relevantPaths) {
      // Navegar pela estrutura do sumário para encontrar as páginas corretas
      for (const secao of sumario) {
        if (path.secao && normalizeStr(secao.secao) !== normalizeStr(path.secao)) continue;
        
        for (const cat of secao.categorias || []) {
          if (path.categoria && normalizeStr(cat.categoria) !== normalizeStr(path.categoria)) continue;
          
          // Se não especificou tópico, pega todas as páginas da categoria
          if (!path.topico) {
            (cat.paginas || []).forEach(p => pagesSet.add(p));
            continue;
          }
          
          for (const top of cat.topicos || []) {
            if (path.topico && normalizeStr(top.topico) !== normalizeStr(path.topico)) continue;
            
            // Se não especificou subtópico, pega todas as páginas do tópico
            if (!path.subtopico) {
              (top.paginas || []).forEach(p => pagesSet.add(p));
              continue;
            }
            
            for (const sub of top.subtopicos || []) {
              if (path.subtopico && normalizeStr(sub.titulo) !== normalizeStr(path.subtopico)) continue;
              (sub.paginas || []).forEach(p => pagesSet.add(p));
            }
          }
        }
      }
    }

    const pages = Array.from(pagesSet).sort((a, b) => a - b);
    
    logSection("Busca Semântica no Sumário - Resultado Final");
    logObj("pages_found", pages);
    logObj("count", pages.length);
    logObj("reasoning_summary", relevantPaths.map(p => ({
      path: `${p.categoria || 'ALL'} > ${p.topico || 'ALL'} > ${p.subtopico || 'ALL'}`,
      reason: p.reasoning
    })));

    return { pages, paths: relevantPaths };
    
  } catch (e) {
    logSection("Busca Semântica no Sumário - Erro");
    logObj("error", String(e));
    return { pages: [], paths: [] };
  }
}

// ==== Funções de Dicionário ====
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

// ==== Funções de HTML ====
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ 
    "&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;" 
  }[c]));
}

function escapeAttr(s) {
  return String(s).replace(/"/g, "&quot;");
}

function buttonForType(tipoRaw, isPremium) {
  const tipo = String(tipoRaw || "").toLowerCase();
  if (tipo.includes("podteme")) return { label: "🎧 Ouvir episódio", kind: "primary" };
  if (tipo.includes("preparatório teme") || tipo.includes("preparatorio teme")) return { label: "▶️ Assistir aula", kind: "accent" };
  if (tipo.includes("instagram")) return { label: "📱 Ver post", kind: "primary" };
  if (tipo.includes("blog")) return { label: "📰 Ler artigo", kind: "primary" };
  if (tipo.includes("curso")) return { label: isPremium ? "💎 Conhecer o curso" : "▶️ Acessar curso", kind: isPremium ? "premium" : "accent" };
  return { label: "🔗 Acessar conteúdo", kind: isPremium ? "premium" : "primary" };
}

function btnStyle(kind) {
  const base = "display:inline-block;padding:8px 12px;border-radius:8px;text-decoration:none;font-weight:500;font-size:13px;border:1px solid;cursor:pointer;";
  if (kind === "accent") return base + "background:rgba(56,189,248,0.08);border-color:rgba(56,189,248,0.25);color:#38bdf8;";
  if (kind === "premium") return base + "background:rgba(245,158,11,0.08);border-color:rgba(245,158,11,0.25);color:#f59e0b;";
  return base + "background:rgba(34,197,94,0.08);border-color:rgba(34,197,94,0.25);color:#22c55e;";
}

function renderDictItemsList(items, isPremiumSection) {
  if (!items.length) return "";
  
  const itemsHtml = items.map(it => {
    const titulo = escapeHtml(it.titulo || "");
    const autor = it.autor ? ` <span style="color:#94a3b8">— ${escapeHtml(it.autor)}</span>` : "";
    const tipo = it.tipoConteudo || it.tipo_conteudo || "";
    const { label, kind } = buttonForType(tipo, !!it.pago);
    const href = it.link ? ` href="${escapeAttr(it.link)}" target="_blank"` : "";
    const btn = it.link ? `<div style="margin-top:6px"><a style="${btnStyle(kind)}"${href}>${label}</a></div>` : "";
    
    const badges = isPremiumSection ? 
      `<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px"><span style="border:1px dashed #1f2937;border-radius:999px;padding:4px 8px;font-size:11px;color:#94a3b8">Carga horária: 12h</span><span style="border:1px dashed #1f2937;border-radius:999px;padding:4px 8px;font-size:11px;color:#94a3b8">Aulas on-demand</span><span style="border:1px dashed #1f2937;border-radius:999px;padding:4px 8px;font-size:11px;color:#94a3b8">Certificado</span></div>` : "";
    
    return `<div style="padding:10px;border:1px solid #1f2937;border-radius:8px;background:rgba(255,255,255,0.015);margin-bottom:8px"><div><strong>${titulo}</strong>${autor}</div>${btn}${badges}</div>`;
  }).join("");
  
  const color = isPremiumSection ? "#f59e0b" : "#22c55e";
  const label = isPremiumSection ? "Conteúdo premium (opcional)" : "Conteúdo complementar";
  
  return `<section style="background:linear-gradient(180deg,#0b1220,#111827);border:1px solid #1f2937;border-radius:12px;padding:14px;margin-bottom:12px"><span style="display:inline-flex;align-items:center;gap:6px;padding:5px 9px;border-radius:999px;border:1px solid #1f2937;background:rgba(255,255,255,0.02);color:#cbd5e1;font-weight:600;font-size:11px;letter-spacing:0.3px;text-transform:uppercase"><span style="width:6px;height:6px;border-radius:50%;background:${color}"></span>${label}</span><div style="margin-top:10px">${itemsHtml}</div></section>`;
}

function renderFinalHtml({ bookAnswer, citedPages, dictItems }) {
  const header = `<header style="margin-bottom:14px"><h1 style="font-size:18px;margin:0 0 6px 0;font-weight:600;color:#1a1a1a">Encontrei a informação que responde à sua dúvida 👇</h1></header>`;

  const bookSection = `<section style="background:linear-gradient(180deg,#0b1220,#111827);border:1px solid #1f2937;border-radius:12px;padding:14px;margin-bottom:12px"><span style="display:inline-flex;align-items:center;gap:6px;padding:5px 9px;border-radius:999px;border:1px solid #1f2937;background:rgba(255,255,255,0.02);color:#cbd5e1;font-weight:600;font-size:11px;letter-spacing:0.3px;text-transform:uppercase"><span style="width:6px;height:6px;border-radius:50%;background:#38bdf8"></span>Livro (fonte principal)</span><div style="position:relative;padding:12px 14px;border-left:3px solid #38bdf8;background:rgba(56,189,248,0.06);border-radius:6px;line-height:1.5;margin-top:10px"><div>${escapeHtml(bookAnswer).replace(/\n/g, "<br>")}</div><small style="display:block;color:#94a3b8;margin-top:6px;font-size:11px">Trechos do livro-base do curso.</small></div></section>`;

  const freeItems = (dictItems || []).filter(x => !x.pago);
  const premiumItems = (dictItems || []).filter(x => x.pago);
  
  let content = header + bookSection;
  if (freeItems.length) content += renderDictItemsList(freeItems, false);
  if (premiumItems.length) content += renderDictItemsList(premiumItems, true);

  return `<div style="max-width:680px;font-family:system-ui,-apple-system,sans-serif;color:#e5e7eb">${content}</div>`;
}

// ==== Recomendação de Dicionário ====
async function recommendFromDictionary(req, question) {
  try {
    const baseUrl = buildBaseUrl(req);
    const res = await fetch(`${baseUrl}/api/dict`);
    if (!res.ok) throw new Error(`GET /api/dict falhou: ${res.status}`);
    const dictItems = await res.json();
    if (!Array.isArray(dictItems) || dictItems.length === 0) return { raw: [] };

    logSection("Dicionário - total carregado");
    logObj("count", dictItems.length);

    const candidates = pickTopDictCandidates(dictItems, question, DICT_MAX_CANDIDATES);
    logSection("Dicionário - candidatos enviados ao modelo");
    logObj("candidates_count", candidates.length);

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

// ==== Transcrição de Áudio ====
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

// ==== Configuração para Next.js ====
export const config = {
  api: { bodyParser: { sizeLimit: "25mb" } }
};

// ==== HANDLER PRINCIPAL (COM BUSCA SEMÂNTICA NO SUMÁRIO) ====
export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();

  // Habilita logs sempre
  als.enterWith({ logs: [], enabled: true });
  const getLogs = () => (als.getStore()?.logs || []);

  try {
    // Processar entrada (texto ou áudio)
    const { question: questionRaw, audio, audio_mime } = req.body || {};
    let question = String(questionRaw || "").trim();
    
    if (!question && audio) {
      logSection("Entrada de áudio detectada");
      question = await transcribeBase64AudioToText(audio, audio_mime || "audio/webm");
    }
    
    if (!question || !question.trim()) {
      return res.status(400).json({ error: "Pergunta vazia", logs: getLogs() });
    }

    logSection("Pergunta recebida");
    logObj("question", question);

    // ============================================
    // 1️⃣ CARREGA DADOS
    // ============================================
    logSection("Etapa 1: Carregamento de dados");
    const [bookRaw, embRaw, sumRaw] = await Promise.all([
      fs.readFile(BOOK_PATH, "utf8"),
      fs.readFile(EMB_PATH, "utf8"),
      fs.readFile(SUM_PATH, "utf8")
    ]);
    const pages = JSON.parse(bookRaw);
    const pageEmbeddings = JSON.parse(embRaw);
    const sumario = JSON.parse(sumRaw);

    const pageMap = new Map(pages.map(p => [p.pagina, p.texto]));
    const embByPage = new Map(pageEmbeddings.map(pe => [pe.pagina, pe.embedding]));

    logObj("pages_loaded", pages.length);
    logObj("embeddings_loaded", pageEmbeddings.length);
    logObj("sumario_sections", sumario.length);

    // ============================================
    // 2️⃣ BUSCA SEMÂNTICA NO SUMÁRIO (COM OPENAI)
    // ============================================
    logSection("Etapa 2: Busca semântica no sumário");
    const summaryResult = await semanticSearchSummary(sumario, question);
    const pagesFromSummary = summaryResult.pages || [];
    const relevantPaths = summaryResult.paths || [];

    // ============================================
    // 3️⃣ DEFINE ESCOPO DE CANDIDATOS (ANTES DO EMBEDDING)
    // ============================================
    let candidatePages;
    let searchScope = "global";
    
    if (pagesFromSummary.length > 0) {
      // Expande páginas do sumário com contexto adjacente mais amplo (±3 páginas)
      // ✅ Quando temos confiança semântica, podemos incluir mais contexto
      const expandedSet = new Set();
      for (const p of pagesFromSummary) {
        expandedSet.add(p);
        [-3, -2, -1, 1, 2, 3].forEach(offset => {
          const adjacent = p + offset;
          if (pageMap.has(adjacent)) expandedSet.add(adjacent);
        });
      }
      
      candidatePages = Array.from(expandedSet)
        .filter(p => embByPage.has(p))
        .sort((a, b) => a - b);
      
      searchScope = "scoped";
      
      logSection("Etapa 3: Escopo restrito por sumário semântico");
      logObj("strategy", "busca focada em capítulos identificados por IA");
      logObj("relevant_paths_found", relevantPaths.length);
      logObj("candidatePages", candidatePages);
      logObj("count", candidatePages.length);
      
    } else {
      // Fallback: busca global se sumário não encontrou nada
      candidatePages = pageEmbeddings
        .map(pe => pe.pagina)
        .filter(p => pageMap.has(p));
      
      searchScope = "global";
      
      logSection("Etapa 3: Escopo global (fallback)");
      logObj("strategy", "sumário semântico não retornou resultados - busca global");
      logObj("candidatePages_count", candidatePages.length);
    }

    // ============================================
    // 4️⃣ SÓ AGORA GERA EMBEDDING DA PERGUNTA
    // ============================================
    logSection("Etapa 4: Geração de embedding (escopo já definido)");
    const embReq = { model: EMB_MODEL, input: question };
    logOpenAIRequest("embeddings.create", embReq);
    
    const tEmb0 = Date.now();
    const qEmbResp = await openai.embeddings.create(embReq);
    const embMs = Date.now() - tEmb0;
    
    logOpenAIResponse("embeddings.create", qEmbResp, {
      duration_ms: embMs,
      embedding_dim: qEmbResp.data?.[0]?.embedding?.length,
      search_scope: searchScope,
      candidates_to_compare: candidatePages.length
    });
    
    const queryEmb = qEmbResp.data[0].embedding;

    // ============================================
    // 5️⃣ CALCULA SIMILARIDADE (APENAS NOS CANDIDATOS)
    // ============================================
    logSection("Etapa 5: Cálculo de similaridade (otimizado)");
    
    const qNorm = normalizeStr(question);
    const qTokens = Array.from(
      new Set(qNorm.split(/\W+/).filter(t => t && t.length > 2))
    );

    let minEmb = Infinity, maxEmb = -Infinity, maxLex = 0;
    const prelim = [];
    
    for (const pg of candidatePages) {
      const peEmb = embByPage.get(pg);
      if (!peEmb) continue;
      
      const embScore = cosineSim(queryEmb, peEmb);
      const raw = pageMap.get(pg) || "";
      const txt = normalizeStr(raw);
      
      let lexScore = 0;
      for (const t of qTokens) {
        lexScore += countOccurrences(txt, t);
      }
      
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

    logObj("comparisons_made", prelim.length);
    logObj("efficiency_gain", searchScope === "scoped" 
      ? `${((1 - candidatePages.length / pageEmbeddings.length) * 100).toFixed(1)}% menos comparações`
      : "busca completa necessária");

    // ============================================
    // 6️⃣ RANKING FINAL
    // ============================================
    logSection("Etapa 6: Ranking final");
    
    const ranked = prelim.map(r => {
      const embNorm = (r.embScore - minEmb) / (Math.max(1e-8, maxEmb - minEmb));
      const lexNorm = maxLex > 0 ? r.lexScore / maxLex : 0;

      // Boost para páginas identificadas pelo sumário semântico
      const summaryBoost = r.inSummary 
        ? (searchScope === "scoped" ? 0.3 : 0.08) 
        : 0;

      // Peso ajustado: embedding mais importante quando escopo já foi filtrado
      const embWeight = searchScope === "scoped" ? 0.8 : 0.7;
      const lexWeight = 1 - embWeight;
      
      const finalScore = embWeight * embNorm + lexWeight * lexNorm + summaryBoost;
      
      return { ...r, embNorm, lexNorm, finalScore };
    }).sort((a, b) => {
      if (b.finalScore !== a.finalScore) return b.finalScore - a.finalScore;
      return a.pagina - b.pagina;
    });

    if (!ranked.length) {
      return res.json({
        answer: "Não encontrei conteúdo no livro.",
        used_pages: [],
        search_scope: searchScope,
        semantic_paths: relevantPaths,
        question_used: question,
        logs: getLogs()
      });
    }

    if (ranked.length) {
      const top10 = ranked.slice(0, Math.min(10, ranked.length));
      logObj("top_10_pages_ranked", top10.map(r => ({
        pagina: r.pagina,
        finalScore: r.finalScore.toFixed(3),
        embScore: r.embScore.toFixed(3),
        lexScore: r.lexScore,
        inSummary: r.inSummary
      })));
      
      logSection("Top 3 para referência rápida");
      logObj("top_3_summary", top10.slice(0, 3).map(r => ({
        pagina: r.pagina,
        score: r.finalScore.toFixed(3)
      })));
    }

    // ============================================
    // 7️⃣ SELECIONA E EXPANDE PÁGINAS
    // ============================================
    logSection("Etapa 7: Seleção e expansão de páginas");
    
    // ✅ Aumentado para TOP 4 páginas para melhor cobertura
    const selectedPages = ranked.slice(0, Math.min(TOP_PAGES_TO_SELECT, ranked.length)).map(r => r.pagina);
    
    let finalPages;
    if (EXPAND_CONTEXT) {
      finalPages = expandWithAdjacentPages(selectedPages, pageMap, ADJACENT_RANGE);
      
      logObj("original_pages", selectedPages);
      logObj("expanded_pages", finalPages);
      logObj("adjacent_range", ADJACENT_RANGE);
    } else {
      finalPages = selectedPages;
    }
    
    const nonEmptyPages = finalPages.filter(p => (pageMap.get(p) || "").trim());
    
    if (!nonEmptyPages.length) {
      return res.json({
        answer: "Não encontrei conteúdo no livro.",
        used_pages: [],
        search_scope: searchScope,
        semantic_paths: relevantPaths,
        question_used: question,
        logs: getLogs()
      });
    }
    
    logObj("final_pages_for_context", nonEmptyPages);
    logObj("total_pages", nonEmptyPages.length);

    // ============================================
    // 8️⃣ MONTA CONTEXTO
    // ============================================
    logSection("Etapa 8: Montagem de contexto");
    
    const contextText = nonEmptyPages.map(p =>
      `--- Página ${p} ---\n${(pageMap.get(p) || "").trim()}\n`
    ).join("\n");
    
    logObj("context_length", contextText.length);
    logObj("context_preview", truncate(contextText, 1000));
    
    // ✅ Verificação de qualidade do contexto
    const qNormCheck = normalizeStr(question);
    const contextNorm = normalizeStr(contextText);
    const qTokensInContext = qTokens.filter(token => contextNorm.includes(token));
    
    logSection("Qualidade do contexto");
    logObj("query_tokens", qTokens);
    logObj("tokens_found_in_context", qTokensInContext);
    logObj("coverage", `${((qTokensInContext.length / qTokens.length) * 100).toFixed(0)}%`);
    
    if (qTokensInContext.length < qTokens.length * 0.3) {
      logLine("⚠️ AVISO: Baixa cobertura de tokens da pergunta no contexto. Resposta pode ser imprecisa.");
    }

    // ============================================
    // 9️⃣ PROMPT E GERAÇÃO
    // ============================================
    logSection("Etapa 9: Geração de resposta");
    
    const systemInstruction = `
Você é um assistente que responde EXCLUSIVAMENTE com trechos literais de um livro-base.

Regras obrigatórias:
- SEMPRE considere que a pergunta é referente a adultos, ou seja, ignore conteúdo pediátrico se não foi solicitado.
- NÃO explique, NÃO resuma, NÃO interprete, NÃO altere palavras, NÃO sintetize.
- Responda SOMENTE com recortes LITERAIS e EXATOS extraídos do livro fornecido.
- Copie cada trecho exatamente como está escrito no texto original, palavra por palavra.
- Identifique cada trecho com o número da página (ex: "- Página 694: \"trecho literal...\"").
- Se houver múltiplos trechos relevantes em páginas diferentes, liste todos.
- NÃO adicione frases introdutórias, comentários, conexões ou resumos.
- Se não houver trechos claramente relevantes, responda apenas "Nenhum trecho encontrado no livro.".

Formato obrigatório da resposta:
- Página N: "recorte literal exato do livro"
- Página M: "outro recorte literal exato do livro"
`.trim();

    const userPrompt = `
Pergunta: """${question}"""

Trechos disponíveis do livro (cada um contém número da página):
${nonEmptyPages.map(p => `Página ${p}:\n${pageMap.get(p)}`).join("\n\n")}

Com base APENAS nos trechos acima, recorte os trechos exatos que respondem diretamente à pergunta.
`.trim();

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

    const answer = chatResp.choices?.[0]?.message?.content?.trim() || "Não encontrei conteúdo no livro.";

    logSection("Resposta bruta gerada");
    logObj("answer", answer);

    // ============================================
    // 🔟 RECOMENDAÇÃO DO DICIONÁRIO
    // ============================================
    logSection("Etapa 10: Recomendação do dicionário");
    const dictRec = await recommendFromDictionary(req, question);

    // ============================================
    // 1️⃣1️⃣ RENDERIZAÇÃO FINAL
    // ============================================
    logSection("Etapa 11: Renderização final");
    
    const notFound = answer === "Não encontrei conteúdo no livro.";
    const citedPages = extractCitedPages(answer);

    const finalAnswer = notFound
      ? answer
      : renderFinalHtml({ bookAnswer: answer, citedPages, dictItems: dictRec.raw });

    logObj("final_output", {
      has_book_answer: !notFound,
      dict_items_count: dictRec.raw.length,
      cited_pages: citedPages
    });

    // ============================================
    // RESPOSTA FINAL
    // ============================================
    return res.status(200).json({
      answer: finalAnswer,
      used_pages: nonEmptyPages,
      original_pages: selectedPages,
      expanded_context: EXPAND_CONTEXT,
      search_scope: searchScope,
      semantic_paths: relevantPaths.map(p => ({
        path: `${p.secao} > ${p.categoria} > ${p.topico || 'ALL'} > ${p.subtopico || 'ALL'}`,
        reasoning: p.reasoning
      })),
      efficiency_metrics: {
        candidates_evaluated: candidatePages.length,
        total_pages: pageEmbeddings.length,
        reduction_percentage: searchScope === "scoped" 
          ? `${((1 - candidatePages.length / pageEmbeddings.length) * 100).toFixed(1)}%`
          : "0%"
      },
      question_used: question,
      logs: getLogs()
    });

  } catch (err) {
    console.error("Erro no /api/chat:", err);
    return res.status(500).json({ 
      error: String(err), 
      logs: getLogs() 
    });
  }
}