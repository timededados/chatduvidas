export function expandWithAdjacentPages(selectedPages, pageMap, range = 0) {
	const expandedSet = new Set();
	for (const page of selectedPages) {
		expandedSet.add(page);
		for (let i = 1; i <= range; i++) {
			const prevPage = page - i;
			if (pageMap.has(prevPage)) expandedSet.add(prevPage);
		}
		for (let i = 1; i <= range; i++) {
			const nextPage = page + i;
			if (pageMap.has(nextPage)) expandedSet.add(nextPage);
		}
	}
	return Array.from(expandedSet).sort((a, b) => a - b);
}
