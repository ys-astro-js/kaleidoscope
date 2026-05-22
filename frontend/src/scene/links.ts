import type { Track } from "../types";
import { SECONDARY_SIMILAR_LIMIT } from "./constants";

export function createSecondarySimilarityLinks(
  selected: Track,
  primaryTracks: Track[],
  tracksById: Map<string, Track>
): Array<{ source: Track; target: Track }> {
  const links: Array<{ source: Track; target: Track }> = [];
  const seenPairs = new Set<string>();

  for (const source of primaryTracks) {
    let addedForSource = 0;
    for (const similar of source.similar) {
      if (similar.id === selected.id || similar.id === source.id) {
        continue;
      }

      const target = tracksById.get(similar.id);
      if (!target) {
        continue;
      }

      const key = [source.id, target.id].sort().join(":");
      if (seenPairs.has(key)) {
        continue;
      }

      seenPairs.add(key);
      links.push({ source, target });
      addedForSource += 1;
      if (addedForSource === SECONDARY_SIMILAR_LIMIT) {
        break;
      }
    }
  }

  return links;
}
