// api/chat.js
import fs from "fs/promises";
import path from "path";

import { als, getLogs, logSection, logObj, logLine, logOpenAIRequest, logOpenAIResponse, truncate } from "../lib/logging.js";
import { normalizeStr, countOccurrences, extractCitedPages, seedFromString, escapeHtml, escapeAttr } from "../lib/text.js";
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

// Delays entre etapas (ms)
const DELAY_AFTER_BOOK_MS = 1500;            // atraso entre "Livro" e "Conteúdo complementar"
const DELAY_AFTER_COMPLEMENTARY_MS = 1500;   // atraso entre "Conteúdo complementar" e "Conteúdo premium"

// Helper de espera
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

// === Helpers locais para renderização parcial (para SSE) ===
function buttonForType(tipoRaw, isPremium) {
	const tipo = String(tipoRaw || "").toLowerCase();
	if (tipo.includes("podteme")) return { label: "🎧 Ouvir episódio", kind: "primary" };
	if (tipo.includes("preparatório teme") || tipo.includes("preparatorio teme")) return { label: "▶️ Assistir aula", kind: "accent" };
	if (tipo.includes("instagram")) return { label: "📱 Ver post", kind: "primary" };
	if (tipo.includes("blog")) return { label: "📰 Ler artigo", kind: "primary" };
	if (tipo.includes("curso")) return { label: isPremium ? "💎 Conhecer o curso" : "▶️ Acessar curso", kind: isPremium ? "premium" : "accent" };
	return { label: "🔗 Acessar conteúdo", kind: isPremium ? "premium" : "primary" };
}
function btnStyle(kind) {
	const base = "display:inline-block;padding:8px 12px;border-radius:8px;text-decoration:none;font-weight:500;font-size:13px;border:1px solid;cursor:pointer;";
	if (kind === "accent") return base + "background:rgba(56,189,248,0.08);border-color:rgba(56,189,248,0.25);color:#38bdf8;";
	if (kind === "premium") return base + "background:rgba(245,158,11,0.08);border-color:rgba(245,158,11,0.25);color:#f59e0b;";
	return base + "background:rgba(34,197,94,0.08);border-color:rgba(34,197,94,0.25);color:#22c55e;";
}
function renderDictItemsList(items, isPremiumSection) {
	if (!items.length) return "";
	const itemsHtml = items.map(it => {
		const titulo = escapeHtml(it.titulo || "");
		const autor = it.autor ? ` <span style="color:#94a3b8">— ${escapeHtml(it.autor)}</span>` : "";
		const tipo = it.tipoConteudo || it.tipo_conteudo || "";
		const { label, kind } = buttonForType(tipo, !!it.pago);
		const href = it.link ? ` href="${escapeAttr(it.link)}" target="_blank"` : "";
		const btn = it.link ? `<div style="margin-top:6px"><a style="${btnStyle(kind)}"${href}>${label}</a></div>` : "";
		const badges = isPremiumSection
			? `<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px"><span style="border:1px dashed #1f2937;border-radius:999px;padding:4px 8px;font-size:11px;color:#94a3b8">Carga horária: 12h</span><span style="border:1px dashed #1f2937;border-radius:999px;padding:4px 8px;font-size:11px;color:#94a3b8">Aulas on-demand</span><span style="border:1px dashed #1f2937;border-radius:999px;padding:4px 8px;font-size:11px;color:#94a3b8">Certificado</span></div>`
			: "";
		return `<div style="padding:10px;border:1px solid #1f2937;border-radius:8px;background:rgba(255,255,255,0.015);margin-bottom:8px"><div><strong>${titulo}</strong>${autor}</div>${btn}${badges}</div>`;
	}).join("");
	const color = isPremiumSection ? "#f59e0b" : "#22c55e";
	const label = isPremiumSection ? "Conteúdo premium (opcional)" : "Conteúdo complementar";
	return `<section style="background:linear-gradient(180deg,#0b1220,#111827);border:1px solid #1f2937;border-radius:12px;padding:14px;margin-bottom:12px"><span style="display:inline-flex;align-items:center;gap:6px;padding:5px 9px;border-radius:999px;border:1px solid #1f2937;background:rgba(255,255,255,0.02);color:#cbd5e1;font-weight:600;font-size:11px;letter-spacing:0.3px;text-transform:uppercase"><span style="width:6px;height:6px;border-radius:50%;background:${color}"></span>${label}</span><div style="margin-top:10px">${itemsHtml}</div></section>`;
}
function renderBookHtml(bookAnswer) {
	const header = `<header style="margin-bottom:14px"><h1 style="font-size:18px;margin:0 0 6px 0;font-weight:600;color:#1a1a1a">Encontrei a informação que responde à sua dúvida 👇</h1></header>`;
	const bookSection = `<section style="background:linear-gradient(180deg,#0b1220,#111827);border:1px solid #1f2937;border-radius:12px;padding:14px;margin-bottom:12px"><span style="display:inline-flex;align-items:center;gap:6px;padding:5px 9px;border-radius:999px;border:1px solid #1f2937;background:rgba(255,255,255,0.02);color:#cbd5e1;font-weight:600;font-size:11px;letter-spacing:0.3px;text-transform:uppercase"><span style="width:6px;height:6px;border-radius:50%;background:#38bdf8"></span>Livro (fonte principal)</span><div style="position:relative;padding:12px 14px;border-left:3px solid #38bdf8;background:rgba(56,189,248,0.06);border-radius:6px;line-height:1.5;margin-top:10px"><div>${escapeHtml(bookAnswer).replace(/\n/g, "<br>")}</div><small style="display:block;color:#94a3b8;margin-top:6px;font-size:11px">Trechos do livro-base do curso.</small></div></section>`;
	return `<div style="max-width:680px;font-family:system-ui,-apple-system,sans-serif;color:#e5e7eb">${header + bookSection}</div>`;
}
function renderDictSection(items, isPremium) {
	if (!items || !items.length) return "";
	return `<div style="max-width:680px;font-family:system-ui,-apple-system,sans-serif;color:#e7e7eb">${renderDictItemsList(items, isPremium)}</div>`;
}

// Next.js API config
export const config = {
	api: { bodyParser: { sizeLimit: "25mb" } }
};

export default async function handler(req, res) {
	if (req.method !== "POST") return res.status(405).end();

	// Habilita logs para a requisição
	als.enterWith({ logs: [], enabled: true });

	try {
		// Detecta modo streaming (SSE)
		const wantsSSE = Boolean(req.body?.stream)
			|| (typeof req.query?.stream !== "undefined" && String(req.query.stream).toLowerCase() !== "false")
			|| String(req.headers?.accept || "").includes("text/event-stream");

		// Helper para SSE (com flush)
		const sse = wantsSSE ? (event, data) => {
			try {
				res.write(`event: ${event}\n`);
				res.write(`data: ${JSON.stringify(data)}\n\n`);
				try { res.flush?.(); } catch {}
			} catch {}
		} : null;

		if (wantsSSE) {
			res.writeHead(200, {
				"Content-Type": "text/event-stream; charset=utf-8",
				"Cache-Control": "no-cache, no-transform",
				"Connection": "keep-alive",
				"X-Accel-Buffering": "no",
				"Access-Control-Allow-Origin": "*"
			});
			// Mantém o socket ativo e sem Nagle
			try { res.socket?.setKeepAlive?.(true); } catch {}
			try { res.socket?.setNoDelay?.(true); } catch {}
			// força envio imediato de headers
			try { res.flushHeaders?.(); } catch {}
			// Padding para furar buffers intermediários
			try {
				res.write(":" + " ".repeat(2048) + "\n\n");
				res.write("event: ready\ndata: {}\n\n");
				try { res.flush?.(); } catch {}
			} catch {}
			// Heartbeat para manter a conexão viva
			const heartbeat = setInterval(() => {
				try {
					res.write(": ping\n\n");
					try { res.flush?.(); } catch {}
				} catch {}
			}, 15000);
			res.on("close", () => {
				try { clearInterval(heartbeat); } catch {}
			});
		}

		// 0) Entrada (texto ou áudio)
		const { question: questionRaw, audio, audio_mime } = req.body || {};
		let question = String(questionRaw || "").trim();
		if (!question && audio) {
			logSection("Entrada de áudio detectada");
			if (wantsSSE) sse("typing", { phase: "book" });
			question = await transcribeBase64AudioToText(audio, audio_mime || "audio/webm");
		}
		if (!question || !question.trim()) {
			if (wantsSSE) {
				sse("error", { message: "Pergunta vazia" });
				sse("done", { logs: getLogs() });
				return res.end();
			}
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
			// Streaming: envia primeiro o livro, depois o dicionário com delays
			if (wantsSSE) {
				const answer = "Não encontrei conteúdo no livro.";
				// Livro (primeiro chunk)
				sse("book", {
					html: renderBookHtml(answer),
					used_pages: [],
					original_pages: []
				});
				try { res.flush?.(); } catch {}

				// Dispara busca do dicionário em paralelo enquanto mostra "digitando..."
				const dictPromise = recommendFromDictionary(req, question);

				// Complementar (typing + delay)
				sse("typing", { phase: "complementary" });
				try { res.flush?.(); } catch {}
				await sleep(DELAY_AFTER_BOOK_MS);

				const dictRec = await dictPromise;
				const freeItems = (dictRec.raw || []).filter(x => !x.pago);
				const premiumItems = (dictRec.raw || []).filter(x => x.pago);

				sse("complementary", {
					html: renderDictSection(freeItems, false),
					count: freeItems.length
				});
				try { res.flush?.(); } catch {}

				// Premium (typing + delay)
				sse("typing", { phase: "premium" });
				try { res.flush?.(); } catch {}
				await sleep(DELAY_AFTER_COMPLEMENTARY_MS);

				sse("premium", {
					html: renderDictSection(premiumItems, true),
					count: premiumItems.length
				});
				try { res.flush?.(); } catch {}

				// Done
				sse("done", {
					search_scope: searchScope,
					semantic_paths: relevantPaths,
					efficiency_metrics: {
						candidates_evaluated: candidatePages.length,
						total_pages: pageEmbeddings.length,
						reduction_percentage: searchScope === "scoped"
							? `${((1 - candidatePages.length / pageEmbeddings.length) * 100).toFixed(1)}%`
							: "0%",
						context_limited: false,
						pages_removed: 0
					},
					question_used: question,
					logs: getLogs()
				});
				return res.end();
			}
			// JSON (comportamento atual)
			return res.json({
				answer: "Não encontrei conteúdo no livro.",
				used_pages: [],
				search_scope: searchScope,
				semantic_paths: relevantPaths,
				question_used: question,
				logs: getLogs()
			});
		}

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
			if (wantsSSE) {
				const answer = "Não encontrei conteúdo no livro.";

				// Livro (primeiro chunk)
				sse("book", {
					html: renderBookHtml(answer),
					used_pages: [],
					original_pages: selectedPages || []
				});
				try { res.flush?.(); } catch {}

				// Dispara busca do dicionário em paralelo
				const dictPromise = recommendFromDictionary(req, question);

				// Complementar (typing + delay)
				sse("typing", { phase: "complementary" });
				try { res.flush?.(); } catch {}
				await sleep(DELAY_AFTER_BOOK_MS);

				const dictRec = await dictPromise;
				const freeItems = (dictRec.raw || []).filter(x => !x.pago);
				const premiumItems = (dictRec.raw || []).filter(x => x.pago);

				sse("complementary", {
					html: renderDictSection(freeItems, false),
					count: freeItems.length
				});
				try { res.flush?.(); } catch {}

				// Premium (typing + delay)
				sse("typing", { phase: "premium" });
				try { res.flush?.(); } catch {}
				await sleep(DELAY_AFTER_COMPLEMENTARY_MS);

				sse("premium", {
					html: renderDictSection(premiumItems, true),
					count: premiumItems.length
				});
				try { res.flush?.(); } catch {}

				sse("done", {
					search_scope: searchScope,
					semantic_paths: relevantPaths,
					efficiency_metrics: {
						candidates_evaluated: candidatePages.length,
						total_pages: pageEmbeddings.length,
						reduction_percentage: searchScope === "scoped"
							? `${((1 - candidatePages.length / pageEmbeddings.length) * 100).toFixed(1)}%`
							: "0%",
						context_limited: false,
						pages_removed: 0
					},
					question_used: question,
					logs: getLogs()
				});
				return res.end();
			}
			// JSON (comportamento atual)
			return res.json({
				answer: "Não encontrei conteúdo no livro.",
				used_pages: [],
				search_scope: searchScope,
				semantic_paths: relevantPaths,
				question_used: question,
				logs: getLogs()
			});
		}

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
		if (wantsSSE) sse("typing", { phase: "book" });
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

		// Livro primeiro no SSE
		if (wantsSSE) {
			sse("book", {
				html: renderBookHtml(answer),
				used_pages: limitedPages,
				original_pages: selectedPages
			});
			try { res.flush?.(); } catch {}

			// Dispara busca do dicionário em paralelo enquanto aguardamos o delay
			const dictPromise = recommendFromDictionary(req, question);

			// Complementar (typing + delay)
			sse("typing", { phase: "complementary" });
			try { res.flush?.(); } catch {}
			await sleep(DELAY_AFTER_BOOK_MS);

			const dictRec = await dictPromise;
			const freeItems = (dictRec.raw || []).filter(x => !x.pago);
			const premiumItems = (dictRec.raw || []).filter(x => x.pago);

			sse("complementary", {
				html: renderDictSection(freeItems, false),
				count: freeItems.length
			});
			try { res.flush?.(); } catch {}

			// Premium (typing + delay)
			sse("typing", { phase: "premium" });
			try { res.flush?.(); } catch {}
			await sleep(DELAY_AFTER_COMPLEMENTARY_MS);

			sse("premium", {
				html: renderDictSection(premiumItems, true),
				count: premiumItems.length
			});
			try { res.flush?.(); } catch {}

			// Done + meta
			sse("done", {
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
			return res.end();
		}

		// 10/11) JSON (comportamento atual)
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
		// ...existing code...
		if (req.body?.stream || String(req.headers?.accept || "").includes("text/event-stream")) {
			try {
				res.write(`event: error\ndata: ${JSON.stringify({ message: String(err) })}\n\n`);
				res.write(`event: done\ndata: ${JSON.stringify({ logs: getLogs() })}\n\n`);
			} catch {}
			return res.end();
		}
		return res.status(500).json({
			error: String(err),
			logs: getLogs()
		});
	}
}