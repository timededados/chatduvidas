import { CHAT_MODEL, DICT_MAX_CANDIDATES, DICT_MAX_RECOMMEND } from "./constants.js";
import { normalizeStr, countOccurrences } from "./text.js";
import { seedFromString } from "./text.js";
import { logSection, logObj, logOpenAIRequest, logOpenAIResponse } from "./logging.js";
import { openai } from "./openai.js";

export function buildBaseUrl(req) {
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

export async function recommendFromDictionary(req, question) {
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
