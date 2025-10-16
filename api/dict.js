import fs from "fs/promises";
import path from "path";
import { createClient } from "@supabase/supabase-js";

const DATA_DIR = path.join(process.cwd(), "data");
const DICT_PATH = path.join(DATA_DIR, "dictionary.json");
const BUCKET_NAME = "dictionary-images"; // bucket público para imagens

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_ANON_KEY
);

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

  // Novo: imagemUrl (opcional)
  const imagemUrl = String(input.imagemUrl || input.imagem_url || "").trim();
  if (imagemUrl && !/^https?:\/\/\S+/i.test(imagemUrl)) {
    return { ok: false, error: "imagemUrl inválida" };
  }

  return { ok: true, value: { titulo, autor, tipo_conteudo: tipoConteudo, pago, link, tags, imagem_url: imagemUrl } };
}

// Upload helper: salva base64 no Storage e retorna public URL
async function uploadBase64ToStorage({ id, base64, mime, originalName }) {
  if (!base64 || !mime || !id) return null;
  if (!/^image\//i.test(mime)) throw new Error("Tipo de imagem inválido");
  const buffer = Buffer.from(base64, "base64");
  const maxBytes = 5 * 1024 * 1024; // 5MB no servidor
  if (buffer.length > maxBytes) throw new Error("Imagem excede 5MB");

  const safeName = String(originalName || `img-${Date.now()}`).replace(/[^a-z0-9._-]+/gi, "-");
  const ext = path.extname(safeName) || "";
  const fname = `${Date.now()}-${safeName}`;
  const objectPath = `${id}/${fname}`;

  const { error: uploadError } = await supabase
    .storage
    .from(BUCKET_NAME)
    .upload(objectPath, buffer, { contentType: mime, upsert: true });

  if (uploadError) throw uploadError;

  const { data: publicData } = supabase.storage.from(BUCKET_NAME).getPublicUrl(objectPath);
  return publicData?.publicUrl || null;
}

export default async function handler(req, res) {
  const { method } = req;
  const { id } = req.query;

  try {
    // GET /api/dict - lista todos
    if (method === "GET" && !id) {
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

      return res.status(200).json(items);
    }

    // GET /api/dict?id=xxx - busca um
    if (method === "GET" && id) {
      const { data, error } = await supabase
        .from("dictionary")
        .select("*")
        .eq("id", id)
        .single();

      if (error) {
        if (error.code === "PGRST116") {
          return res.status(404).json({ error: "Não encontrado" });
        }
        throw error;
      }

      const item = {
        id: data.id,
        titulo: data.titulo,
        autor: data.autor,
        tipoConteudo: data.tipo_conteudo,
        pago: data.pago,
        link: data.link,
        tags: data.tags || [],
        imagemUrl: data.imagem_url || null,
        createdAt: data.created_at,
        updatedAt: data.updated_at
      };

      return res.status(200).json(item);
    }

    // POST /api/dict - criar
    if (method === "POST") {
      const v = validateEntry(req.body || {});
      if (!v.ok) return res.status(400).json({ error: v.error });

      const newId = genId();

      // Se veio imagem em base64, subir para o Storage
      let finalImagemUrl = v.value.imagem_url || null;
      if (req.body?.imagemData) {
        finalImagemUrl = await uploadBase64ToStorage({
          id: newId,
          base64: req.body.imagemData,
          mime: req.body.imagemType,
          originalName: req.body.imagemName
        });
      }

      const insertPayload = {
        id: newId,
        ...v.value,
        imagem_url: finalImagemUrl
      };

      const { data, error } = await supabase
        .from("dictionary")
        .insert([insertPayload])
        .select()
        .single();

      if (error) throw error;

      const result = {
        id: data.id,
        titulo: data.titulo,
        autor: data.autor,
        tipoConteudo: data.tipo_conteudo,
        pago: data.pago,
        link: data.link,
        tags: data.tags || [],
        imagemUrl: data.imagem_url || null,
        createdAt: data.created_at,
        updatedAt: data.updated_at
      };

      return res.status(201).json(result);
    }

    // PUT /api/dict?id=xxx - atualizar
    if (method === "PUT" && id) {
      const v = validateEntry(req.body || {});
      if (!v.ok) return res.status(400).json({ error: v.error });

      // Se veio nova imagem em base64, sobrescreve a URL com a nova
      let finalImagemUrl = v.value.imagem_url || null;
      if (req.body?.imagemData) {
        finalImagemUrl = await uploadBase64ToStorage({
          id,
          base64: req.body.imagemData,
          mime: req.body.imagemType,
          originalName: req.body.imagemName
        });
      }

      const { data, error } = await supabase
        .from("dictionary")
        .update({
          ...v.value,
          imagem_url: finalImagemUrl,
          updated_at: new Date().toISOString()
        })
        .eq("id", id)
        .select()
        .single();

      if (error) {
        if (error.code === "PGRST116") {
          return res.status(404).json({ error: "Não encontrado" });
        }
        throw error;
      }

      const result = {
        id: data.id,
        titulo: data.titulo,
        autor: data.autor,
        tipoConteudo: data.tipo_conteudo,
        pago: data.pago,
        link: data.link,
        tags: data.tags || [],
        imagemUrl: data.imagem_url || null,
        createdAt: data.created_at,
        updatedAt: data.updated_at
      };

      return res.status(200).json(result);
    }

    // DELETE /api/dict?id=xxx - excluir
    if (method === "DELETE" && id) {
      const { error } = await supabase
        .from("dictionary")
        .delete()
        .eq("id", id);

      if (error) throw error;

      return res.status(200).json({ ok: true });
    }

    return res.status(405).json({ error: "Método não permitido" });
  } catch (e) {
    console.error("Erro em /api/dict:", e);
    return res.status(500).json({ error: String(e.message || e) });
  }
}
