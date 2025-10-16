/**
 * server.js
 * Node.js + Express server que:
 * - carrega o livro JSON (abramede_texto.json)
 * - carrega (ou gera) embeddings por página (text-embedding-3-small)
 * - recebe perguntas do frontend, pede ao OpenAI variações da pergunta,
 *   faz busca por similaridade (cosine) e retorna resposta do modelo
 *   pedindo EXPRESSAMENTE para usar apenas o conteúdo fornecido e citar páginas.
 *
 * Ajustes possíveis:
 * - Mudar caminhos dos arquivos em DATA_DIR
 * - Ajustar TOP_K, modelo de chat, etc.
 */

import express from "express";
import bodyParser from "body-parser";
import cors from "cors";
import fs from "fs/promises";
import path from "path";
import OpenAI from "openai";
import { toFile } from "openai/uploads";

const PORT = process.env.PORT || 3000;
const DATA_DIR = path.join(process.cwd(), "data");
const BOOK_PATH = path.join(DATA_DIR, "abramede_texto.json");
const EMB_PATH = path.join(DATA_DIR, "abramede_embeddings.json"); // opcional
// Adicionado: arquivo do dicionário
const DICT_PATH = path.join(DATA_DIR, "dictionary.json");
const TOP_K = 6; // quantas páginas pegar por busca
const EMB_MODEL = "text-embedding-3-small";
const CHAT_MODEL = "gpt-4o-mini"; // você pode trocar (por exemplo gpt-4o)
const MAX_CONTEXT_TOKENS = 3000; // corte para segurança

if (!process.env.OPENAI_API_KEY) {
  console.error("Defina OPENAI_API_KEY no ambiente.");
  process.exit(1);
}
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const app = express();
// Aumenta limite para aceitar áudio em base64/data URL (~MBs)
app.use(bodyParser.json({ limit: "25mb" }));
app.use(bodyParser.urlencoded({ extended: true, limit: "25mb" }));
app.use(cors());

// Utilitário: transcrever áudio base64 para texto (gpt-4o-mini-transcribe)
async function transcribeBase64AudioToText(audioStr, mime = "audio/webm", openaiClient) {
  try {
    const clean = String(audioStr || "").replace(/^data:.*;base64,/, "");
    const buf = Buffer.from(clean, "base64");
    const ext = mime.includes("mpeg") ? "mp3"
      : mime.includes("wav") ? "wav"
      : mime.includes("ogg") ? "ogg"
      : mime.includes("m4a") ? "m4a"
      : "webm";
    const filename = `audio.${ext}`;
    const file = await toFile(buf, filename, { type: mime });
    const resp = await openaiClient.audio.transcriptions.create({
      model: "gpt-4o-mini-transcribe",
      file,
      language: "pt"
    });
    const text = (resp && (resp.text || resp.transcript || resp?.results?.[0]?.transcript)) || "";
    return text.trim();
  } catch (e) {
    console.error("Falha ao transcrever áudio:", e);
    return "";
  }
}

// Adicionado: utilitários para dicionário
async function ensureDataDir() {
  await fs.mkdir(DATA_DIR, { recursive: true });
}
async function loadDictionary() {
  try {
    const raw = await fs.readFile(DICT_PATH, "utf8");
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) throw new Error("Formato inválido");
    return arr;
  } catch {
    return [];
  }
}
async function saveDictionary(arr) {
  await ensureDataDir();
  await fs.writeFile(DICT_PATH, JSON.stringify(arr, null, 2), "utf8");
}
function genId() {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}
function validateEntry(input) {
  const titulo = String(input.titulo || "").trim();
  if (!titulo) return { ok: false, error: "titulo é obrigatório" };
  const autor = String(input.autor || "").trim();
  const tipoConteudo = String(input.tipoConteudo || input["tipo de conteudo"] || "").trim();
  let pagoRaw = input.pago;
  if (typeof pagoRaw === "string") {
    const l = pagoRaw.toLowerCase();
    pagoRaw = l === "sim" || l === "true" || l === "1";
  }
  const pago = Boolean(pagoRaw);
  const link = String(input.link || "").trim();
  if (link && !/^https?:\/\/\S+/i.test(link)) return { ok: false, error: "link inválido" };
  let tags = input.tags;
  if (typeof tags === "string") {
    tags = tags.split(",").map(t => t.trim()).filter(Boolean);
  }
  if (!Array.isArray(tags)) tags = [];
  return { ok: true, value: { titulo, autor, tipoConteudo, pago, link, tags } };
}

// Adicionado: endpoints CRUD do dicionário (ANTES do express.static)
app.get("/api/dict", async (req, res) => {
  try {
    const items = await loadDictionary();
    res.json(items);
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

app.get("/api/dict/:id", async (req, res) => {
  try {
    const items = await loadDictionary();
    const found = items.find(i => i.id === req.params.id);
    if (!found) return res.status(404).json({ error: "Não encontrado" });
    res.json(found);
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

app.post("/api/dict", async (req, res) => {
  try {
    const v = validateEntry(req.body || {});
    if (!v.ok) return res.status(400).json({ error: v.error });
    const items = await loadDictionary();
    const now = new Date().toISOString();
    const item = { id: genId(), createdAt: now, updatedAt: now, ...v.value };
    items.push(item);
    await saveDictionary(items);
    res.status(201).json(item);
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

app.put("/api/dict/:id", async (req, res) => {
  try {
    const items = await loadDictionary();
    const idx = items.findIndex(i => i.id === req.params.id);
    if (idx === -1) return res.status(404).json({ error: "Não encontrado" });
    const v = validateEntry(req.body || {});
    if (!v.ok) return res.status(400).json({ error: v.error });
    const now = new Date().toISOString();
    items[idx] = { ...items[idx], ...v.value, updatedAt: now };
    await saveDictionary(items);
    res.json(items[idx]);
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

app.delete("/api/dict/:id", async (req, res) => {
  try {
    const items = await loadDictionary();
    const exists = items.some(i => i.id === req.params.id);
    if (!exists) return res.status(404).json({ error: "Não encontrado" });
    const filtered = items.filter(i => i.id !== req.params.id);
    await saveDictionary(filtered);
    res.json({ ok: true });
  } catch (e) {
    res.status(500).json({ error: String(e) });
  }
});

// endpoint principal do chat
app.post("/api/chat", async (req, res) => {
  try {
    // Novo: aceitar pergunta por voz (base64/data URL)
    const { question: questionRaw, audio, audio_mime } = req.body || {};
    if (audio) {
      // Log leve para depuração de tamanho e tipo de mídia
      try { console.log("Audio payload:", { mime: audio_mime, len: String(audio).length }); } catch {}
    }
    let question = String(questionRaw || "").trim();
    if (!question && audio) {
      question = await transcribeBase64AudioToText(audio, audio_mime || "audio/webm", openai);
    }
    if (!question || !question.trim()) return res.status(400).json({ error: "Pergunta vazia" });

    // 1) Primeiro peça ao modelo variações da pergunta para melhorar busca (few-shot)
    const variationPrompt = `
Você é um assistente que ajuda a gerar variações de consulta de busca para localizar trechos em um livro.
Dada a pergunta do usuário, gere até 6 variações curtas (1-12 palavras cada) que mantenham o sentido,
incluindo sinônimos e sinônimos técnicos quando for o caso. Retorne um JSON com campo "variations": [ ... ].
Pergunta: """${question}"""
Se já está ótima, devolva ao menos a pergunta original como primeira variação.
`;
    const varResp = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages: [
        { role: "system", content: "Você gera variações de consultas para busca sem adicionar conteúdo novo." },
        { role: "user", content: variationPrompt }
      ],
      max_tokens: 300,
      temperature: 0.2
    });
    const varText = varResp.choices?.[0]?.message?.content?.trim() || "";
    // tentativa robusta de extrair JSON; se falhar, usar fallback
    let variations = [];
    try {
      const maybeJson = varText.match(/\{[\s\S]*\}/);
      if (maybeJson) {
        const parsed = JSON.parse(maybeJson[0]);
        if (Array.isArray(parsed.variations)) variations = parsed.variations;
      }
    } catch (e) {
      // fallback: pegar linhas
      variations = varText.split(/\r?\n/).map(l => l.trim()).filter(Boolean).slice(0, 6);
    }
    if (!variations.length) variations = [question];

    // 2) Carregar livro e embeddings
    const pages = await loadBook(); // [{pagina, texto}]
    let pageEmbeddings = await loadEmbeddings();
    if (!pageEmbeddings) {
      pageEmbeddings = await generateEmbeddingsForPages(pages);
    }

    // 3) Gerar embedding para cada variação e buscar (você pode combinar scores)
    const variationEmbeddingsPromises = variations.map(v =>
      openai.embeddings.create({ model: EMB_MODEL, input: v })
    );
    const variationEmbResps = await Promise.all(variationEmbeddingsPromises);
    const varEmbeddings = variationEmbResps.map(r => r.data[0].embedding);

    // para cada variação, pega top results; depois consolida por score (soma)
    const aggregate = new Map(); // pagina -> aggregatedScore
    for (let i = 0; i < varEmbeddings.length; i++) {
      const emb = varEmbeddings[i];
      const results = await semanticSearch(emb, pageEmbeddings, pages, TOP_K);
      for (const r of results) {
        const prev = aggregate.get(r.pagina) || 0;
        // weight pelo rank/posição também: usar score diretamente (já cosine)
        aggregate.set(r.pagina, prev + r.score);
      }
    }
    // ordena páginas por score agregado
    const aggArr = Array.from(aggregate.entries()).map(([pagina, score]) => ({ pagina, score }));
    aggArr.sort((a, b) => b.score - a.score);
    const selected = aggArr.slice(0, TOP_K).map(a => {
      const p = pages.find(x => x.pagina === a.pagina);
      return { pagina: a.pagina, texto: (p && p.texto) || "", score: a.score };
    });

    // 4) Monta contexto para o modelo de resposta: inclui apenas as páginas selecionadas.
    // Cuidado com tamanho: truncar se necessário.
    let contextBuilder = [];
    let totalLen = 0;
    for (const s of selected) {
      const snippet = (s.texto || "").trim();
      // estimativa simples de tokens = ~4 chars per token (very rough). We'll cap by chars.
      const charLen = snippet.length;
      if (totalLen + charLen > MAX_CONTEXT_TOKENS * 4) break;
      contextBuilder.push(`--- Página ${s.pagina} ---\n${snippet}\n`);
      totalLen += charLen;
    }
    const contextText = contextBuilder.join("\n");

    // If no meaningful content found -> reply negative
    if (!contextText.trim()) {
      return res.json({ answer: "Não encontrei conteúdo no livro.", question_used: question });
    }

    // 5) Pergunta final ao modelo pedindo resposta estrita somente com o conteúdo.
    const systemInstruction = `
Você é um assistente que responde perguntas EXCLUSIVAMENTE com base no conteúdo fornecido a seguir.
Não invente informações. Se a resposta não puder ser determinada a partir do conteúdo, responda exatamente:
"Não encontrei conteúdo no livro."
Ao responder, seja sucinto e inclua entre parênteses a página usada para cada afirmação, por exemplo (p. 123).
Use somente os trechos fornecidos e cite páginas.`;
    const userPrompt = `
Conteúdo do livro (apenas trechos abaixo). Use apenas esse conteúdo:

${contextText}

Pergunta do usuário: """${question}"""
Responda em português.
`;

    const chatResp = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages: [
        { role: "system", content: systemInstruction },
        { role: "user", content: userPrompt }
      ],
      temperature: 0.0,
      max_tokens: 800
    });

    const answer = chatResp.choices?.[0]?.message?.content?.trim() || "Não encontrei conteúdo no livro.";
    return res.json({
      answer,
      used_pages: selected.map(s => ({ pagina: s.pagina, score: s.score })),
      question_used: question
    });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: String(err) });
  }
});

// IMPORTANTE: express.static deve vir DEPOIS das rotas da API
app.use(express.static("public"));

// Adicionado: fallback JSON para /api (evita HTML em erros)
app.use("/api", (req, res) => {
  res.status(404).json({ error: `Rota não encontrada: ${req.method} ${req.originalUrl}` });
});

app.listen(PORT, () => console.log(`Server rodando em http://localhost:${PORT}`));
