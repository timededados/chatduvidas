// api/chat.js
import fs from "fs/promises";
import path from "path";

import { als, getLogs, logSection, logObj, logLine, logOpenAIRequest, logOpenAIResponse, truncate } from "../lib/logging.js";
import { normalizeStr, countOccurrences, extractCitedPages, seedFromString } from "../lib/text.js";
import { cosineSim } from "../lib/similarity.js";
import { expandWithAdjacentPages } from "../lib/context.js";
import { semanticSearchSummary } from "../lib/summarySearch.js";
import { recommendFromDictionary } from "../lib/dict.js";
import { renderFinalHtml } from "../lib/render.js";
import { transcribeBase64AudioToText } from "../lib/transcription.js";
import { openai } from "../lib/openai.js";
import { EMB_MODEL, CHAT_MODEL } from "../lib/constants.js";

// Caminhos de dados (mantidos)
const DATA_DIR = path.join(process.cwd(), "data");
const BOOK_PATH = path.join(DATA_DIR, "abramede_texto.json");
const EMB_PATH = path.join(DATA_DIR, "abramede_embeddings.json");
const SUM_PATH = path.join(DATA_DIR, "sumario_final.json");

// Parâmetros de busca/contexto (mantidos)
const EXPAND_CONTEXT = false;
const ADJACENT_RANGE = 0;
const TOP_PAGES_TO_SELECT = 9;

// Next.js API config
export const config = {
	api: { bodyParser: { sizeLimit: "25mb" } }
};

export default async function handler(req, res) {
	if (req.method !== "POST") return res.status(405).end();

	// Habilita logs para a requisição
	als.enterWith({ logs: [], enabled: true });

	try {
		// 0) Entrada (texto ou áudio)
		const { question: questionRaw, audio, audio_mime } = req.body || {};
		let question = String(questionRaw || "").trim();
		if (!question && audio) {
			logSection("Entrada de áudio detectada");
			question = await transcribeBase64AudioToText(audio, audio_mime || "audio/webm");
		}
		if (!question || !question.trim()) {
			return res.status(400).json({ error: "Pergunta vazia", logs: getLogs() });
		}
		logSection("Pergunta recebida");
		logObj("question", question);

		// 1) Carregamento de dados
		logSection("Etapa 1: Carregamento de dados");
		const [bookRaw, embRaw, sumRaw] = await Promise.all([
			fs.readFile(BOOK_PATH, "utf8"),
			fs.readFile(EMB_PATH, "utf8"),
			fs.readFile(SUM_PATH, "utf8")
		]);
		const pages = JSON.parse(bookRaw);
		const pageEmbeddings = JSON.parse(embRaw);
		const sumario = JSON.parse(sumRaw);

		const pageMap = new Map(pages.map(p => [p.pagina, p.texto]));
		const embByPage = new Map(pageEmbeddings.map(pe => [pe.pagina, pe.embedding]));
		logObj("pages_loaded", pages.length);
		logObj("embeddings_loaded", pageEmbeddings.length);
		logObj("sumario_sections", sumario.length);

		// 2) Busca semântica no sumário
		logSection("Etapa 2: Busca semântica no sumário");
		const summaryResult = await semanticSearchSummary(sumario, question);
		const pagesFromSummary = summaryResult.pages || [];
		const relevantPaths = summaryResult.paths || [];

		// 3) Define escopo de candidatos
		let candidatePages;
		let searchScope = "global";
		if (pagesFromSummary.length > 0) {
			const expandedSet = new Set();
			for (const p of pagesFromSummary) {
				expandedSet.add(p);
				[-2, -1, 1, 2].forEach(offset => {
					const adjacent = p + offset;
					if (pageMap.has(adjacent)) expandedSet.add(adjacent);
				});
			}
			candidatePages = Array.from(expandedSet)
				.filter(p => embByPage.has(p))
				.sort((a, b) => a - b);
			searchScope = "scoped";
			logSection("Etapa 3: Escopo restrito por sumário semântico");
			logObj("relevant_paths_found", relevantPaths.length);
			logObj("candidatePages", candidatePages);
			logObj("count", candidatePages.length);
		} else {
			candidatePages = pageEmbeddings
				.map(pe => pe.pagina)
				.filter(p => pageMap.has(p));
			searchScope = "global";
			logSection("Etapa 3: Escopo global (fallback)");
			logObj("candidatePages_count", candidatePages.length);
		}

		// 4) Embedding da pergunta
		logSection("Etapa 4: Geração de embedding (escopo já definido)");
		const embReq = { model: EMB_MODEL, input: question };
		logOpenAIRequest("embeddings.create", embReq);
		const tEmb0 = Date.now();
		const qEmbResp = await openai.embeddings.create(embReq);
		const embMs = Date.now() - tEmb0;
		logOpenAIResponse("embeddings.create", qEmbResp, {
			duration_ms: embMs,
			embedding_dim: qEmbResp.data?.[0]?.embedding?.length,
			search_scope: searchScope,
			candidates_to_compare: candidatePages.length
		});
		const queryEmb = qEmbResp.data[0].embedding;

		// 5) Similaridade (apenas no escopo)
		logSection("Etapa 5: Cálculo de similaridade (otimizado)");
		const qNorm = normalizeStr(question);
		const qTokens = Array.from(new Set(qNorm.split(/\W+/).filter(t => t && t.length > 2)));
		let minEmb = Infinity, maxEmb = -Infinity, maxLex = 0;
		const prelim = [];
		for (const pg of candidatePages) {
			const peEmb = embByPage.get(pg);
			if (!peEmb) continue;
			const embScore = cosineSim(queryEmb, peEmb);
			const raw = pageMap.get(pg) || "";
			const txt = normalizeStr(raw);
			let lexScore = 0;
			for (const t of qTokens) lexScore += countOccurrences(txt, t);
			prelim.push({ pagina: pg, embScore, lexScore, inSummary: pagesFromSummary.includes(pg) });
			if (embScore < minEmb) minEmb = embScore;
			if (embScore > maxEmb) maxEmb = embScore;
			if (lexScore > maxLex) maxLex = lexScore;
		}
		logObj("comparisons_made", prelim.length);
		logObj("efficiency_gain", searchScope === "scoped"
			? `${((1 - candidatePages.length / pageEmbeddings.length) * 100).toFixed(1)}% menos comparações`
			: "busca completa necessária");

		// 6) Ranking final
		logSection("Etapa 6: Ranking final");
		const ranked = prelim.map(r => {
			const embNorm = (r.embScore - minEmb) / (Math.max(1e-8, maxEmb - minEmb));
			const lexNorm = maxLex > 0 ? r.lexScore / maxLex : 0;
			const summaryBoost = r.inSummary ? (searchScope === "scoped" ? 0.3 : 0.08) : 0;
			const embWeight = searchScope === "scoped" ? 0.8 : 0.7;
			const lexWeight = 1 - embWeight;
			const finalScore = embWeight * embNorm + lexWeight * lexNorm + summaryBoost;
			return { ...r, embNorm, lexNorm, finalScore };
		}).sort((a, b) => (b.finalScore - a.finalScore) || (a.pagina - b.pagina));

		if (!ranked.length) {
			return res.json({
				answer: "Não encontrei conteúdo no livro.",
				used_pages: [],
				search_scope: searchScope,
				semantic_paths: relevantPaths,
				question_used: question,
				logs: getLogs()
			});
		}

		const top10 = ranked.slice(0, Math.min(10, ranked.length));
		logSection("Top 3 para referência rápida");
		logObj("top_3_summary", top10.slice(0, 3).map(r => ({ pagina: r.pagina, score: r.finalScore.toFixed(3) })));

		// 7) Seleção e expansão de páginas
		logSection("Etapa 7: Seleção e expansão de páginas");
		const selectedPages = ranked.slice(0, Math.min(TOP_PAGES_TO_SELECT, ranked.length)).map(r => r.pagina);
		const finalPages = EXPAND_CONTEXT
			? expandWithAdjacentPages(selectedPages, pageMap, ADJACENT_RANGE)
			: selectedPages;

		const nonEmptyPages = finalPages.filter(p => (pageMap.get(p) || "").trim());
		const MAX_CONTEXT_PAGES = 10;
		let limitedPages = nonEmptyPages;
		if (nonEmptyPages.length > MAX_CONTEXT_PAGES) {
			const pagesWithScore = nonEmptyPages.map(p => {
				const rankInfo = ranked.find(r => r.pagina === p);
				return { pagina: p, score: rankInfo ? rankInfo.finalScore : 0 };
			}).sort((a, b) => b.score - a.score);
			limitedPages = pagesWithScore.slice(0, MAX_CONTEXT_PAGES).map(p => p.pagina).sort((a, b) => a - b);
			logSection("⚠️ Contexto limitado por tamanho");
			logObj("original_count", nonEmptyPages.length);
			logObj("limited_to", limitedPages.length);
			logObj("removed_pages", nonEmptyPages.filter(p => !limitedPages.includes(p)));
		}
		if (!limitedPages.length) {
			return res.json({
				answer: "Não encontrei conteúdo no livro.",
				used_pages: [],
				search_scope: searchScope,
				semantic_paths: relevantPaths,
				question_used: question,
				logs: getLogs()
			});
		}
		logObj("final_pages_for_context", limitedPages);
		logObj("total_pages", limitedPages.length);

		// 8) Contexto
		logSection("Etapa 8: Montagem de contexto");
		const contextText = limitedPages.map(p => `--- Página ${p} ---\n${(pageMap.get(p) || "").trim()}\n`).join("\n");
		logObj("context_length", contextText.length);
		logObj("context_preview", truncate(contextText, 1000));

		const qNormCheck = normalizeStr(question);
		const contextNorm = normalizeStr(contextText);
		const qTokensInContext = Array.from(new Set(qNormCheck.split(/\W+/).filter(t => t && t.length > 2)))
			.filter(token => contextNorm.includes(token));
		logSection("Qualidade do contexto");
		logObj("tokens_found_in_context", qTokensInContext);

		if (qTokensInContext.length < Math.max(1, qTokens.length * 0.3)) {
			logLine("⚠️ AVISO: Baixa cobertura de tokens da pergunta no contexto. Resposta pode ser imprecisa.");
		}

		// 9) Geração (resposta literal)
		logSection("Etapa 9: Geração de resposta");
		const systemInstruction = `
Você é um assistente que responde EXCLUSIVAMENTE com trechos literais de um livro-base.

Regras obrigatórias:
- SEMPRE considere que a pergunta é referente a adultos, ou seja, ignore conteúdo pediátrico se não foi solicitado.
- NÃO explique, NÃO resuma, NÃO interprete, NÃO altere palavras, NÃO sintetize.
- Responda SOMENTE com recortes LITERAIS e EXATOS extraídos do livro fornecido.
- Copie cada trecho exatamente como está escrito no texto original, palavra por palavra.
- Identifique cada trecho com o número da página (ex: "- Página 694: "trecho literal...").
- Se houver múltiplos trechos relevantes em páginas diferentes, liste todos.
- NÃO adicione frases introdutórias, comentários, conexões ou resumos.
- Se não houver trechos claramente relevantes, responda apenas "Nenhum trecho encontrado no livro.".

Formato obrigatório da resposta:
- Página N: "recorte literal exato do livro"
- Página M: "outro recorte literal exato do livro"
`.trim();

		const userPrompt = `
Pergunta: """${question}"""

Trechos disponíveis do livro (cada um contém número da página):
${limitedPages.map(p => `Página ${p}:\n${pageMap.get(p)}`).join("\n\n")}

Com base APENAS nos trechos acima, recorte os trechos exatos que respondem diretamente à pergunta.
`.trim();

		const chatReq = {
			model: CHAT_MODEL,
			messages: [
				{ role: "system", content: systemInstruction },
				{ role: "user", content: userPrompt }
			],
			temperature: 0,
			top_p: 1,
			frequency_penalty: 0,
			presence_penalty: 0,
			n: 1,
			max_tokens: 900,
			seed: seedFromString(question)
		};

		logOpenAIRequest("chat.completions.create", chatReq);
		const tChat0 = Date.now();
		const chatResp = await openai.chat.completions.create(chatReq);
		const chatMs = Date.now() - tChat0;
		logOpenAIResponse("chat.completions.create", chatResp, { duration_ms: chatMs });

		const answer = chatResp.choices?.[0]?.message?.content?.trim() || "Não encontrei conteúdo no livro.";
		logSection("Resposta bruta gerada");
		logObj("answer", answer);

		// 10) Dicionário
		logSection("Etapa 10: Recomendação do dicionário");
		const dictRec = await recommendFromDictionary(req, question);

		// 11) Renderização final
		logSection("Etapa 11: Renderização final");
		const notFound = answer === "Não encontrei conteúdo no livro.";
		const citedPages = extractCitedPages(answer);
		const finalAnswer = notFound ? answer : renderFinalHtml({ bookAnswer: answer, citedPages, dictItems: dictRec.raw });

		logObj("final_output", {
			has_book_answer: !notFound,
			dict_items_count: dictRec.raw.length,
			cited_pages: citedPages
		});

		return res.status(200).json({
			answer: finalAnswer,
			used_pages: limitedPages,
			original_pages: selectedPages,
			expanded_context: EXPAND_CONTEXT,
			search_scope: searchScope,
			semantic_paths: relevantPaths.map(p => ({
				path: `${p.secao} > ${p.categoria} > ${p.topico || 'ALL'} > ${p.subtopico || 'ALL'}`,
				reasoning: p.reasoning
			})),
			efficiency_metrics: {
				candidates_evaluated: candidatePages.length,
				total_pages: pageEmbeddings.length,
				reduction_percentage: searchScope === "scoped"
					? `${((1 - candidatePages.length / pageEmbeddings.length) * 100).toFixed(1)}%`
					: "0%",
				context_limited: nonEmptyPages.length > limitedPages.length,
				pages_removed: nonEmptyPages.length - limitedPages.length
			},
			question_used: question,
			logs: getLogs()
		});
	} catch (err) {
		console.error("Erro no /api/chat:", err);
		return res.status(500).json({
			error: String(err),
			logs: getLogs()
		});
	}
}