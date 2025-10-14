// api/chat.js
import OpenAI from "openai";
import fs from "fs/promises";
import path from "path";

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
const DATA_DIR = path.join(process.cwd(), "data");
const BOOK_PATH = path.join(DATA_DIR, "abramede_texto.json");

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).end();

  try {
    const { question } = req.body;
    if (!question) return res.status(400).json({ error: "Pergunta vazia" });

    const raw = await fs.readFile(BOOK_PATH, "utf8");
    const pages = JSON.parse(raw);

    // Aqui você pode simplificar (sem embeddings, só pra testar)
    const answer = `Recebi sua pergunta: "${question}" e encontrei ${pages.length} páginas no livro.`;

    return res.status(200).json({ answer });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: err.message });
  }
}
