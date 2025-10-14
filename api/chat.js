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

// Busca simples no sumário por similaridade textual
function searchSummary(sumario, query) {
  const normalized = query.toLowerCase();
  const results = [];
  for (const top of sumario) {
    const allText = `${top.topico} ${top.subtopicos
      ?.map(s => s.titulo)
      .join(" ")}`.toLowerCase();
    if (allText.includes(normalized)) {
      results.push({
        topico: top.topico,
        paginas: top.paginas || [],
        subtopicos: top.subtopicos || []
      });
    }
  }
  return results;
}

// Localiza o tópico/subtópico de uma página
function findTopicForPage(sumario, pagina) {
  for (const top of sumario) {
    const topRange = top.paginas || [];
    if (topRange.includes(pagina)) return { topico: top.topico, subt: null };
    for (const s of top.subtopicos || []) {
      if ((s.paginas || []).includes(pagina))
        return { topico: top.topico, subt: s.titulo };
    }
  }
  return null;
}

// ---------- Função principal ----------
export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();

  try {
    const { question } = req.body;
    if (!question || !question.trim())
      return res.status(400).json({ error: "Pergunta vazia" });

    // 1️⃣ Gera variações da pergunta
    const variationPrompt = `
Gere até 6 variações curtas (1-12 palavras) da pergunta a seguir para ajudar na busca de conteúdo técnico.
Responda em JSON: {"variations": ["..."]}

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

    // 2️⃣ Carrega dados
    const [bookRaw, embRaw, sumRaw] = await Promise.all([
      fs.readFile(BOOK_PATH, "utf8"),
      fs.readFile(EMB_PATH, "utf8"),
      fs.readFile(SUM_PATH, "utf8")
    ]);
    const pages = JSON.parse(bookRaw);
    const pageEmbeddings = JSON.parse(embRaw);
    const sumario = JSON.parse(sumRaw);

    const pageMap = new Map(pages.map(p => [p.pagina, p.texto]));

    // 3️⃣ Busca no sumário
    const summaryMatches = searchSummary(sumario, question);
    const pagesFromSummary = summaryMatches.flatMap(m => [
      ...(m.paginas || []),
      ...m.subtopicos.flatMap(s => s.paginas || [])
    ]);

    // 4️⃣ Busca semântica (embeddings)
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

    const sorted = Array.from(aggregate.entries())
      .map(([pagina, score]) => ({ pagina, score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, TOP_K);

    const pagesFromEmbedding = sorted.map(p => p.pagina);

    // 5️⃣ Combina páginas do sumário + embeddings
    const combinedSet = new Set([...pagesFromSummary, ...pagesFromEmbedding]);

    // 6️⃣ Adiciona 1 página anterior e 1 posterior
    const expandedSet = new Set();
    for (const p of combinedSet) {
      expandedSet.add(p);
      if (p > 1) expandedSet.add(p - 1);
      expandedSet.add(p + 1);
    }
    const expandedPages = Array.from(expandedSet).sort((a, b) => a - b);

    // 7️⃣ Monta o contexto
    let contextBuilder = [];
    let totalLen = 0;
    for (const pagina of expandedPages) {
      const snippet = (pageMap.get(pagina) || "").trim();
      const len = snippet.length;
      if (totalLen + len > MAX_CONTEXT_TOKENS * 4) break;
      contextBuilder.push(`--- Página ${pagina} ---\n${snippet}\n`);
      totalLen += len;
    }

    const contextText = contextBuilder.join("\n");
    if (!contextText.trim())
      return res.json({ answer: "Não encontrei conteúdo no livro." });

    // 8️⃣ Prompt restritivo com citações literais e negrito
    const systemInstruction = `
Você é um assistente que responde perguntas exclusivamente com base no texto abaixo.
⚠️ REGRAS IMPORTANTES:
- NÃO adicione informações externas.
- Use APENAS as frases originais do texto fornecido.
- Sempre inclua as citações exatas entre aspas e em negrito ("**texto**"), indicando a página de origem.
- Se a informação não estiver no texto, responda exatamente:
  "Não encontrei conteúdo no livro."
`;

    const userPrompt = `
Conteúdo do livro (trechos das páginas):

${contextText}

Pergunta do usuário:
"""${question}"""

Responda usando SOMENTE as palavras originais do livro, citando entre aspas e em negrito as partes utilizadas e a(s) página(s) correspondente(s).
`;

    const chatResp = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages: [
        { role: "system", content: systemInstruction },
        { role: "user", content: userPrompt }
      ],
      temperature: 0,
      max_tokens: 900
    });

    const rawAnswer =
      chatResp.choices?.[0]?.message?.content?.trim() ||
      "Não encontrei conteúdo no livro.";

    // 9️⃣ Extrai páginas realmente citadas na resposta
    const citedPages = [];
    const regex = /\(\*\*p[aá]gina[s]?\s*(\d+)\*\*\)/gi;
    let match;
    while ((match = regex.exec(rawAnswer)) !== null) {
      const num = parseInt(match[1], 10);
      if (!isNaN(num)) citedPages.push(num);
    }
    const uniquePages = Array.from(new Set(citedPages)).sort((a, b) => a - b);

    // Se o modelo não citou páginas, usamos as expandidas
    const usedPages = uniquePages.length ? uniquePages : expandedPages;

    // Localiza tópico/subtópico do primeiro número citado
    const ref = findTopicForPage(sumario, usedPages[0]) || {};
    const topicName = ref.topico || "tópico não identificado";
    const subName = ref.subt || "subtópico não identificado";

    // Define range de páginas (somente das citadas)
    const pageRange =
      usedPages.length === 1
        ? `página ${usedPages[0]}`
        : `páginas ${usedPages[0]}–${usedPages[usedPages.length - 1]}`;

    // Cabeçalho formatado
    const formatted = `Essa resposta pode ser encontrada no tópico "${topicName}", subtópico "${subName}", na(s) ${pageRange}.\n\n**Resposta:**\n${rawAnswer}`;

    return res.status(200).json({
      answer: formatted,
      used_pages: usedPages
    });

  } catch (err) {
    console.error("Erro no /api/chat:", err);
    return res.status(500).json({ error: String(err) });
  }
}
