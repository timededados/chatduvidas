import { openai } from "./openai.js";
import { CHAT_MODEL } from "./constants.js";
import { seedFromString } from "./text.js";
import { logSection, logObj, logOpenAIRequest, logOpenAIResponse, truncate } from "./logging.js";

export async function semanticSearchSummary(sumario, question) {
	try {
		logSection("Busca Semântica no Sumário - Início");

		const flatStructure = [];
		try {
			for (const secao of sumario) {
				for (const cat of secao.categorias || []) {
					if (cat.paginas?.length) {
						flatStructure.push({
							categoria: cat.categoria,
							topico: null,
							subtopico: null,
							paginas: cat.paginas,
							_secao: secao.secao
						});
					}
					for (const top of cat.topicos || []) {
						if (top.paginas?.length) {
							flatStructure.push({
								categoria: cat.categoria,
								topico: top.topico,
								subtopico: null,
								paginas: top.paginas,
								_secao: secao.secao
							});
						}
						for (const sub of top.subtopicos || []) {
							if (sub.paginas?.length) {
								flatStructure.push({
									categoria: cat.categoria,
									topico: top.topico,
									subtopico: sub.titulo,
									paginas: sub.paginas,
									_secao: secao.secao
								});
							}
						}
					}
				}
			}
		} catch (structureError) {
			logSection("Busca Semântica no Sumário - Erro na estruturação");
			logObj("error", String(structureError));
			return { pages: [], paths: [] };
		}

		if (!flatStructure.length) {
			logSection("Busca Semântica no Sumário - Estrutura vazia");
		 return { pages: [], paths: [] };
		}

		const summaryIndex = flatStructure.map((item, idx) => [
			idx,
			item.categoria,
			item.topico || "-",
			item.subtopico || "-",
			(item.paginas || []).length
		]);

		logSection("Busca Semântica no Sumário - Preparando Prompt");
		const systemPrompt = `Você é um especialista em medicina de emergência e terapia intensiva.

Identifique itens do índice médico relevantes para a pergunta.

IMPORTANTE:
- Considere sinônimos (PCR = parada cardiorrespiratória = RCP = ressuscitação cardiopulmonar)
- Considere abreviações (IAM, AVC, TEP, RCP, PCR)
- Procure em categoria, tópico E subtópico
- Seja INCLUSIVO: se a pergunta menciona "RCP", retorne TODAS as ocorrências
- Padrão: adultos (ignore pediátrico a menos que solicitado)

FORMATO DO ÍNDICE (array):
[id, "categoria", "tópico ou -", "subtópico ou -", qtd_páginas]

Responda APENAS JSON:
{"relevant_indices": [0, 5, 12]}

Se nada relevante: {"relevant_indices": []}`;

		const userPrompt = `Pergunta: "${question}"

Índice (${summaryIndex.length} itens):
${JSON.stringify(summaryIndex)}

Retorne os IDs (primeiro número de cada array) dos itens relevantes.`;

		const chatReq = {
			model: CHAT_MODEL,
			messages: [
				{ role: "system", content: systemPrompt },
				{ role: "user", content: userPrompt }
		 ],
			temperature: 0,
			top_p: 1,
			max_tokens: 800,
			seed: seedFromString(question + "|summary")
		};

		logOpenAIRequest("chat.completions.create [semantic_summary]", chatReq);
		const t0 = Date.now();
		const resp = await openai.chat.completions.create(chatReq);
		const ms = Date.now() - t0;
		logOpenAIResponse("chat.completions.create [semantic_summary]", resp, { duration_ms: ms });

		logSection("Busca Semântica no Sumário - Parseando Resposta");
		const raw = resp.choices?.[0]?.message?.content?.trim() || "{}";
		logObj("raw_response_preview", truncate(raw, 500));

		let relevantIndices = [];
		try {
			const m = raw.match(/\{[\s\S]*\}/);
			const parsed = JSON.parse(m ? m[0] : raw);
			relevantIndices = parsed.relevant_indices || [];
		} catch (e) {
			logSection("Busca Semântica no Sumário - Erro ao parsear JSON");
			logObj("parse_error", String(e));
			logObj("raw_response", raw);
			return { pages: [], paths: [] };
		}

		logSection("Busca Semântica no Sumário - Coletando Páginas");
		const pagesSet = new Set();
		const relevantPaths = [];
		try {
			for (const idx of relevantIndices) {
				if (idx >= 0 && idx < flatStructure.length) {
					const item = flatStructure[idx];
					(item.paginas || []).forEach(p => pagesSet.add(p));
					relevantPaths.push({
						secao: item._secao,
						categoria: item.categoria,
						topico: item.topico,
						subtopico: item.subtopico,
						reasoning: `ID ${idx}: ${item.categoria} > ${item.topico || 'GERAL'} > ${item.subtopico || 'GERAL'}`,
						pages_count: (item.paginas || []).length
					});
				}
			}
		} catch (collectionError) {
			logSection("Busca Semântica no Sumário - Erro ao coletar páginas");
			logObj("error", String(collectionError));
			return { pages: [], paths: [] };
		}

		const pages = Array.from(pagesSet).sort((a, b) => a - b);
		logSection("Busca Semântica no Sumário - Resultado Final");
		logObj("items_found", relevantPaths.length);
		logObj("unique_pages", pages.length);

		return { pages, paths: relevantPaths };
	} catch (e) {
		logSection("Busca Semântica no Sumário - Erro");
		logObj("error", String(e));
		return { pages: [], paths: [] };
	}
}
