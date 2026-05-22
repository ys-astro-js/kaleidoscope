import type { SimilarTrack, Track, ViewMode } from "../types";
import { PRIMARY_SIMILAR_LIMIT } from "./constants";
import { createSecondarySimilarityLinks } from "./links";

export function createNodeSignature(tracks: Track[], viewMode: ViewMode): string {
  return tracks
    .map(
      (track) =>
        `${viewMode}:${track.id}:${formatCoord(track.x)}:${formatCoord(track.y)}:${formatCoord(track.z)}`
    )
    .join("|");
}

export function createLinkSignature(
  tracks: Track[],
  selectedId: string | null,
  linkSourceId: string | null,
  similarLinks: SimilarTrack[] | null,
  highlightedLinkId: string | null
): string {
  const source = tracks.find((track) => track.id === (linkSourceId ?? selectedId));
  if (!source) {
    return "";
  }
  if (similarLinks) {
    const segmentSimilar = similarLinks
      .slice(0, PRIMARY_SIMILAR_LIMIT)
      .map((track) => `${track.id}:${track.score.toFixed(4)}`)
      .join(",");
    return `${source.id}:segment:${highlightedLinkId ?? ""}:${segmentSimilar}`;
  }
  const tracksById = new Map(tracks.map((track) => [track.id, track]));
  const similar = source.similar
    .slice(0, PRIMARY_SIMILAR_LIMIT)
    .map((track) => `${track.id}:${track.score.toFixed(4)}`)
    .join(",");
  const primaryTracks = source.similar
    .slice(0, PRIMARY_SIMILAR_LIMIT)
    .map((track) => tracksById.get(track.id))
    .filter((track): track is Track => Boolean(track));
  const secondary = createSecondarySimilarityLinks(source, primaryTracks, tracksById)
    .map((link) => `${link.source.id}>${link.target.id}`)
    .join(",");
  return `${source.id}:${highlightedLinkId ?? ""}:${similar}:${secondary}`;
}

function formatCoord(value: number | null): string {
  return value == null ? "" : value.toFixed(4);
}
