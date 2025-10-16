import fs from "fs/promises";
import path from "path";

export default async function handler(req, res) {
  if (req.method !== "GET") {
    return res.status(405).json({ error: "Método não permitido" });
  }

  try {
    const sumarioPath = path.join(process.cwd(), "data", "sumario_final.json");
    const raw = await fs.readFile(sumarioPath, "utf8");
    const sumario = JSON.parse(raw);
    
    // Extrair todas as categorias únicas
    const categoriasSet = new Set();
    sumario.forEach(secao => {
      (secao.categorias || []).forEach(cat => {
        if (cat.categoria) {
          categoriasSet.add(cat.categoria);
        }
      });
    });
    
    const categorias = Array.from(categoriasSet).sort();
    
    return res.status(200).json({ categorias });
  } catch (e) {
    console.error("Erro ao carregar categorias:", e);
    return res.status(500).json({ error: String(e.message || e) });
  }
}
