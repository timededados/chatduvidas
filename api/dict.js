import fs from "fs/promises";
import path from "path";

const DATA_DIR = path.join(process.cwd(), "data");
const DICT_PATH = path.join(DATA_DIR, "dictionary.json");

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

export default async function handler(req, res) {
  const { method } = req;
  const { id } = req.query;

  try {
    // GET /api/dict - lista todos
    if (method === "GET" && !id) {
      const items = await loadDictionary();
      return res.status(200).json(items);
    }

    // GET /api/dict?id=xxx - busca um
    if (method === "GET" && id) {
      const items = await loadDictionary();
      const found = items.find(i => i.id === id);
      if (!found) return res.status(404).json({ error: "Não encontrado" });
      return res.status(200).json(found);
    }

    // POST /api/dict - criar
    if (method === "POST") {
      const v = validateEntry(req.body || {});
      if (!v.ok) return res.status(400).json({ error: v.error });
      const items = await loadDictionary();
      const now = new Date().toISOString();
      const item = { id: genId(), createdAt: now, updatedAt: now, ...v.value };
      items.push(item);
      await saveDictionary(items);
      return res.status(201).json(item);
    }

    // PUT /api/dict?id=xxx - atualizar
    if (method === "PUT" && id) {
      const items = await loadDictionary();
      const idx = items.findIndex(i => i.id === id);
      if (idx === -1) return res.status(404).json({ error: "Não encontrado" });
      const v = validateEntry(req.body || {});
      if (!v.ok) return res.status(400).json({ error: v.error });
      const now = new Date().toISOString();
      items[idx] = { ...items[idx], ...v.value, updatedAt: now };
      await saveDictionary(items);
      return res.status(200).json(items[idx]);
    }

    // DELETE /api/dict?id=xxx - excluir
    if (method === "DELETE" && id) {
      const items = await loadDictionary();
      const exists = items.some(i => i.id === id);
      if (!exists) return res.status(404).json({ error: "Não encontrado" });
      const filtered = items.filter(i => i.id !== id);
      await saveDictionary(filtered);
      return res.status(200).json({ ok: true });
    }

    return res.status(405).json({ error: "Método não permitido" });
  } catch (e) {
    return res.status(500).json({ error: String(e) });
  }
}
