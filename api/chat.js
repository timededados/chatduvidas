// api/chat.js
import OpenAI from "openai";
import fs from "fs/promises";
import path from "path";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// Caminhos dentro da função serverless (Vercel copia tudo pro root)
const DATA_DIR = path.join(process.cwd(), "data");
const BOOK_PATH = path.join(DATA_DIR, "abramede_texto.json");
const EMB_PATH = path.join(DATA_DIR, "abramede_embeddings.json");

const EMB_MODEL = "text-embedding-3-small";
const CHAT_MODEL = "gpt-4o-mini";
const TOP_K = 6;
const MAX_CONTEXT_TOKENS = 3000;

// utilitários
function dot(a, b) {
  return a.reduce((s, v, i) => s + v * b[i], 0);
}
function norm(a) {
  return Math.sqrt(a.reduce((s, x) => s + x * x, 0));
}
function cosineSim(a, b) {
  return dot(a, b) / (norm(a) * norm(b) + 1e-8);
}

// --- função principal da API ---
export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();

  try {
    const { question } = req.body;
    if (!question || !question.trim())
      return res.status(400).json({ error: "Pergunta vazia" });

    // 🔹 1. Gera variações da pergunta
    const variationPrompt = `
Você é um assistente que ajuda a gerar variações de consulta de busca para localizar trechos em um livro.
Dada a pergunta do usuário, gere até 6 variações curtas (1-12 palavras cada) que mantenham o sentido.
Responda apenas em JSON: {"variations": ["..."]}

Pergunta: """${question}"""`;

    const varResp = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages: [
        { role: "system", content: "Você gera variações de consultas para busca." },
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
      // fallback em caso de erro de JSON
      variations = rawVar.split(/\r?\n/).filter(Boolean).slice(0, 6);
      if (!variations.length) variations = [question];
    }

    // 🔹 2. Carrega livro
    const bookRaw = await fs.readFile(BOOK_PATH, "utf8");
    const pages = JSON.parse(bookRaw);

    // 🔹 3. Carrega embeddings salvos (ou gera na primeira execução)
    let pageEmbeddings;
    try {
      const embRaw = await fs.readFile(EMB_PATH, "utf8");
      pageEmbeddings = JSON.parse(embRaw);
    } catch {
      // gera embeddings (atenção ao custo)
      pageEmbeddings = [];
      for (const p of pages) {
        const txt = (p.texto || "").slice(0, 2000);
        const emb = await openai.embeddings.create({
          model: EMB_MODEL,
          input: txt || " "
        });
        pageEmbeddings.push({ pagina: p.pagina, embedding: emb.data[0].embedding });
      }
      await fs.writeFile(EMB_PATH, JSON.stringify(pageEmbeddings), "utf8");
    }

    // 🔹 4. Busca semântica
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

    // Ordena páginas mais relevantes
    const sorted = Array.from(aggregate.entries())
      .map(([pagina, score]) => ({ pagina, score }))
      .sort((a, b) => b.score - a.score)
      .slice(0, TOP_K);

    const selected = sorted.map(s => {
      const p = pages.find(x => x.pagina === s.pagina);
      return { pagina: s.pagina, texto: p?.texto || "", score: s.score };
    });

    // 🔹 5. Monta contexto para o modelo
    let contextBuilder = [];
    let totalLen = 0;
    for (const s of selected) {
      const snippet = (s.texto || "").trim();
      const len = snippet.length;
      if (totalLen + len > MAX_CONTEXT_TOKENS * 4) break;
      contextBuilder.push(`--- Página ${s.pagina} ---\n${snippet}\n`);
      totalLen += len;
    }

    const contextText = contextBuilder.join("\n");
    if (!contextText.trim())
      return res.json({ answer: "Não encontrei conteúdo no livro." });

    // 🔹 6. Pede resposta com base no livro
    const systemInstruction = `
Você é um assistente que responde perguntas EXCLUSIVAMENTE com base no conteúdo fornecido.
Não invente nada. Se não houver informação suficiente, diga:
"Não encontrei conteúdo no livro."
Cite as páginas utilizadas entre parênteses, ex: (p. 45).
`;

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
      temperature: 0,
      max_tokens: 800
    });

    const answer = chatResp.choices?.[0]?.message?.content?.trim()
      || "Não encontrei conteúdo no livro.";

    return res.status(200).json({
      answer,
      used_pages: selected.map(s => ({ pagina: s.pagina, score: s.score }))
    });

  } catch (err) {
    console.error("Erro no /api/chat:", err);
    return res.status(500).json({ error: String(err) });
  }
}
