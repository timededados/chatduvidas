export function dot(a, b) {
	return a.reduce((s, v, i) => s + v * b[i], 0);
}
export function norm(a) {
	return Math.sqrt(a.reduce((s, x) => s + x * x, 0));
}
export function cosineSim(a, b) {
	return dot(a, b) / (norm(a) * norm(b) + 1e-8);
}
