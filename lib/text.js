export function seedFromString(s) {
	let h = 5381;
	for (let i = 0; i < s.length; i++) h = ((h << 5) + h) + s.charCodeAt(i);
	return Math.abs(h >>> 0);
}

export function normalizeStr(s) {
	return (s || "")
		.toLowerCase()
		.normalize("NFD")
		.replace(/[\u0300-\u036f]/g, "");
}

export function countOccurrences(text, token) {
	if (!token) return 0;
	const re = new RegExp(`\\b${token.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "g");
	return (text.match(re) || []).length;
}

export function extractCitedPages(text) {
	if (!text) return [];
	const set = new Set();
	const patterns = [
		/página\s+(\d+)/gi,
		/pagina\s+(\d+)/gi,
		/\(p\.\s*(\d+)\)/gi
	];
	for (const re of patterns) {
		let m;
		while ((m = re.exec(text)) !== null) {
			const n = parseInt(m[1], 10);
			if (!isNaN(n)) set.add(n);
		}
	}
	return Array.from(set).sort((a, b) => a - b);
}

export function escapeHtml(s) {
	return String(s).replace(/[&<>"']/g, c => ({
		"&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;", "'": "&#39;"
	}[c]));
}

export function escapeAttr(s) {
	return String(s).replace(/"/g, "&quot;");
}
