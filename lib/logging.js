import { AsyncLocalStorage } from "async_hooks";

export const als = new AsyncLocalStorage();
const TRUNC_LIMIT = 800;

export function truncate(str, n = TRUNC_LIMIT) {
	if (typeof str !== "string") return str;
	return str.length > n ? str.slice(0, n) + `... [${str.length - n} more chars]` : str;
}

export function getLogs() {
	return als.getStore()?.logs || [];
}

export function logSection(title) {
	const store = als.getStore();
	if (!(store && store.enabled)) return;
	if (store.logs) store.logs.push(`=== ${title} ===`);
	console.log(`\n=== ${title} ===`);
}

export function logObj(label, obj) {
	const store = als.getStore();
	if (!(store && store.enabled)) return;
	let rendered;
	try { rendered = JSON.stringify(obj, null, 2); }
	catch { rendered = String(obj); }
	if (store.logs) store.logs.push(`${label}: ${rendered}`);
	console.log(label, rendered);
}

export function logLine(...args) {
	const store = als.getStore();
	if (!(store && store.enabled)) return;
	const msg = args.map(a => {
		if (typeof a === "string") return a;
		try { return JSON.stringify(a); } catch { return String(a); }
	}).join(" ");
	store.logs.push(msg);
	console.log(msg);
}

export function logOpenAIRequest(kind, payload) {
	const store = als.getStore();
	if (!(store && store.enabled)) return;
	const clone = { ...payload };
	if (Array.isArray(clone.messages)) {
		clone.messages = clone.messages.map(m => ({
			role: m.role,
			content: truncate(m.content, 600)
		}));
	}
	if (typeof clone.input === "string") clone.input = truncate(clone.input, 600);
	logSection(`Requisição OpenAI: ${kind}`);
	logObj("payload", clone);
}

export function logOpenAIResponse(kind, resp, extra = {}) {
	const store = als.getStore();
	if (!(store && store.enabled)) return;
	const safe = {
		id: resp.id,
		model: resp.model,
		usage: resp.usage,
		created: resp.created,
		choices: (resp.choices || []).map(c => ({
			index: c.index,
			finish_reason: c.finish_reason,
			message: c.message ? {
				role: c.message.role,
				content: truncate(c.message.content, 600)
			} : undefined
		})),
		...extra
	};
	logSection(`Resposta OpenAI: ${kind}`);
	logObj("data", safe);
}
