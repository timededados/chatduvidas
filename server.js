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

const PORT = process.env.PORT || 3000;
const DATA_DIR = path.join(process.cwd(), "data");
const BOOK_PATH = path.join(DATA_DIR, "abramede_texto.json"); // já enviado. :contentReference[oaicite:2]{index=2}
const EMB_PATH = path.join(DATA_DIR, "abramede_embeddings.json"); // opcional
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
app.use(bodyParser.json());
app.use(cors());
app.use(express.static("public"));

// util: cosine similarity
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
function norm(a) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s);
}
function cosineSim(a, b) {
  return dot(a, b) / (norm(a) * norm(b) + 1e-8);
}

// carrega livro JSON
async function loadBook() {
  const raw = await fs.readFile(BOOK_PATH, "utf8");
  const pages = JSON.parse(raw);
  // espera array de objetos {pagina: number, texto: string}
  return pages;
}

// carrega embeddings (se existir). formato esperado: [{pagina: num, embedding: [num,...]}]
async function loadEmbeddings() {
  try {
    const raw = await fs.readFile(EMB_PATH, "utf8");
    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) throw new Error("embeddings inválidos");
    return arr;
  } catch (err) {
    console.warn("Embeddings não encontrados em", EMB_PATH, "- serão gerados sob demanda.");
    return null;
  }
}

// gera embeddings para cada página (cuidado com custo). Retorna array [{pagina, embedding}]
async function generateEmbeddingsForPages(pages) {
  console.log("Gerando embeddings para", pages.length, "páginas (pode demorar e custar chamadas)...");
  const results = [];
  // geramos em batches simples (não otimizado)
  for (const p of pages) {
    const text = (p.texto || "").slice(0, 2000); // truncar texto muito grande para embedding
    const resp = await openai.embeddings.create({
      model: EMB_MODEL,
      input: text || " " // não enviar vazio
    });
    const embedding = resp.data[0].embedding;
    results.push({ pagina: p.pagina, embedding });
  }
  // salva para reuso
  await fs.mkdir(DATA_DIR, { recursive: true });
  await fs.writeFile(EMB_PATH, JSON.stringify(results), "utf8");
  console.log("Embeddings gerados e salvos em", EMB_PATH);
  return results;
}

// busca: recebe query embedding, retorna top K páginas (objetos {pagina, texto, score})
async function semanticSearch(queryEmbedding, pageEmbeddings, pages, topK = TOP_K) {
  const scores = pageEmbeddings.map(pe => {
    return {
      pagina: pe.pagina,
      score: cosineSim(queryEmbedding, pe.embedding)
    };
  });
  scores.sort((a, b) => b.score - a.score);
  const top = scores.slice(0, topK);
  // map to page text
  const pageMap = new Map(pages.map(p => [p.pagina, p.texto]));
  return top.map(t => ({ pagina: t.pagina, texto: pageMap.get(t.pagina) || "", score: t.score }));
}

// endpoint principal do chat
app.post("/api/chat", async (req, res) => {
  try {
    const { question } = req.body;
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
      return res.json({ answer: "Não encontrei conteúdo no livro." });
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
    return res.json({ answer, used_pages: selected.map(s => ({ pagina: s.pagina, score: s.score })) });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: String(err) });
  }
});

app.listen(PORT, () => console.log(`Server rodando em http://localhost:${PORT}`));
