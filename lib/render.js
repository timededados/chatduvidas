import { escapeHtml, escapeAttr } from "./text.js";

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
		const badges = isPremiumSection ?
			`<div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px"><span style="border:1px dashed #1f2937;border-radius:999px;padding:4px 8px;font-size:11px;color:#94a3b8">Carga horária: 12h</span><span style="border:1px dashed #1f2937;border-radius:999px;padding:4px 8px;font-size:11px;color:#94a3b8">Aulas on-demand</span><span style="border:1px dashed #1f2937;border-radius:999px;padding:4px 8px;font-size:11px;color:#94a3b8">Certificado</span></div>` : "";
		return `<div style="padding:10px;border:1px solid #1f2937;border-radius:8px;background:rgba(255,255,255,0.015);margin-bottom:8px"><div><strong>${titulo}</strong>${autor}</div>${btn}${badges}</div>`;
	}).join("");

	const color = isPremiumSection ? "#f59e0b" : "#22c55e";
	const label = isPremiumSection ? "Conteúdo premium (opcional)" : "Conteúdo complementar";

	return `<section style="background:linear-gradient(180deg,#0b1220,#111827);border:1px solid #1f2937;border-radius:12px;padding:14px;margin-bottom:12px"><span style="display:inline-flex;align-items:center;gap:6px;padding:5px 9px;border-radius:999px;border:1px solid #1f2937;background:rgba(255,255,255,0.02);color:#cbd5e1;font-weight:600;font-size:11px;letter-spacing:0.3px;text-transform:uppercase"><span style="width:6px;height:6px;border-radius:50%;background:${color}"></span>${label}</span><div style="margin-top:10px">${itemsHtml}</div></section>`;
}

export function renderFinalHtml({ bookAnswer, citedPages, dictItems }) {
	const header = `<header style="margin-bottom:14px"><h1 style="font-size:18px;margin:0 0 6px 0;font-weight:600;color:#1a1a1a">Encontrei a informação que responde à sua dúvida 👇</h1></header>`;
	const bookSection = `<section style="background:linear-gradient(180deg,#0b1220,#111827);border:1px solid #1f2937;border-radius:12px;padding:14px;margin-bottom:12px"><span style="display:inline-flex;align-items:center;gap:6px;padding:5px 9px;border-radius:999px;border:1px solid #1f2937;background:rgba(255,255,255,0.02);color:#cbd5e1;font-weight:600;font-size:11px;letter-spacing:0.3px;text-transform:uppercase"><span style="width:6px;height:6px;border-radius:50%;background:#38bdf8"></span>Livro (fonte principal)</span><div style="position:relative;padding:12px 14px;border-left:3px solid #38bdf8;background:rgba(56,189,248,0.06);border-radius:6px;line-height:1.5;margin-top:10px"><div>${escapeHtml(bookAnswer).replace(/\n/g, "<br>")}</div><small style="display:block;color:#94a3b8;margin-top:6px;font-size:11px">Trechos do livro-base do curso.</small></div></section>`;

	const freeItems = (dictItems || []).filter(x => !x.pago);
	const premiumItems = (dictItems || []).filter(x => x.pago);

	let content = header + bookSection;
	if (freeItems.length) content += renderDictItemsList(freeItems, false);
	if (premiumItems.length) content += renderDictItemsList(premiumItems, true);

	return `<div style="max-width:680px;font-family:system-ui,-apple-system,sans-serif;color:#e5e7eb">${content}</div>`;
}
