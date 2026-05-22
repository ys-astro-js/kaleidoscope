import type { SimilarSegment, Track } from "./types";

export const SEGMENT_HOP_SECONDS = 15;
export const SEGMENT_WINDOW_SECONDS = 30;
export const CROSSFADE_SECONDS = 4;
export const TRACK_AUTOPLAY_HIGHLIGHT_SECONDS = 5;
export const AUTOPLAY_TRIGGER_SECONDS = SEGMENT_WINDOW_SECONDS - CROSSFADE_SECONDS;
export const AUTOPLAY_HISTORY_LIMIT = 12;

export type SegmentVisit = {
  trackId: string;
  segmentIndex: number;
};

export type RoutePreview = SegmentVisit & {
  matches: SimilarSegment[];
  nextTrackId: string | null;
};

export type TrackAutoplayPreview = SegmentVisit & {
  match: SimilarSegment;
  nextTrackId: string;
  sourceEndSeconds: number;
  triggerSeconds: number;
};

export type TrackAutoplayCandidate = SegmentVisit & {
  match: SimilarSegment;
  track: Track;
  sourceEndSeconds: number;
  triggerSeconds: number;
};

export function segmentIndexForTime(timeSeconds: number): number {
  return Math.max(0, Math.floor(timeSeconds / SEGMENT_HOP_SECONDS));
}

export function segmentWindowIndexForTime(timeSeconds: number): number {
  return segmentIndexForTime(Math.floor(Math.max(0, timeSeconds) / SEGMENT_WINDOW_SECONDS) * SEGMENT_WINDOW_SECONDS);
}

export function withHistoryVisit(history: SegmentVisit[], visit: SegmentVisit): SegmentVisit[] {
  const last = history.at(-1);
  if (last?.trackId === visit.trackId && last.segmentIndex === visit.segmentIndex) {
    return history;
  }
  return [...history, visit].slice(-AUTOPLAY_HISTORY_LIMIT);
}

export function chooseAutoplayMatch(
  matches: SimilarSegment[],
  allTracks: Track[],
  history: SegmentVisit[],
  currentVisit?: SegmentVisit
): { match: SimilarSegment; track: Track } | null {
  const nextHistory = currentVisit ? withHistoryVisit(history, currentVisit) : history;
  const recentSegmentKeys = new Set(
    nextHistory
      .slice(-AUTOPLAY_HISTORY_LIMIT)
      .map((visit) => segmentKey(visit.trackId, visit.segmentIndex))
  );
  const previousVisit = nextHistory.at(-2);

  const candidates = matches
    .map((match) => ({
      match,
      track: allTracks.find((track) => track.id === match.id)
    }))
    .filter(
      (candidate): candidate is { match: SimilarSegment; track: Track } =>
        Boolean(candidate.track)
    )
    .filter(({ match }) => {
      const key = segmentKey(match.id, match.segment_index);
      const isRecentSegment = recentSegmentKeys.has(key);
      const isImmediateBacktrack =
        previousVisit?.trackId === match.id && previousVisit.segmentIndex === match.segment_index;
      return !isRecentSegment && !isImmediateBacktrack;
    });

  return candidates[0] ?? null;
}

export function buildRoutePreview(
  source: SegmentVisit,
  matches: SimilarSegment[],
  allTracks: Track[],
  history: SegmentVisit[],
  includeNext: boolean
): RoutePreview {
  return {
    ...source,
    matches,
    nextTrackId: includeNext
      ? chooseAutoplayMatch(matches, allTracks, history, source)?.track.id ?? null
      : null,
  };
}

export function buildTrackAutoplayCandidates(
  matches: SimilarSegment[],
  allTracks: Track[],
  source: SegmentVisit & { sourceEndSeconds: number; triggerSeconds: number }
): TrackAutoplayCandidate[] {
  return matches
    .map((match) => ({
      ...source,
      match,
      track: allTracks.find((track) => track.id === match.id)
    }))
    .filter(
      (candidate): candidate is TrackAutoplayCandidate =>
        Boolean(candidate.track)
    );
}

export function chooseTrackAutoplayCandidate(
  candidates: TrackAutoplayCandidate[],
  recentTrackIds: string[]
): TrackAutoplayCandidate | null {
  if (candidates.length === 0) {
    return null;
  }
  const recentTracks = new Set(recentTrackIds.slice(-AUTOPLAY_HISTORY_LIMIT));
  const preferredCandidates = candidates.filter(({ track }) => !recentTracks.has(track.id));
  const pool = preferredCandidates.length > 0 ? preferredCandidates : candidates;
  return pool.reduce((best, candidate) =>
    candidate.match.score > best.match.score ? candidate : best
  );
}

export function trackAutoplaySourceSegmentIndexes(
  storedSegmentCount: number,
  duration: number
): number[] {
  if (storedSegmentCount > 0) {
    return Array.from(
      new Set([storedSegmentCount - 1, storedSegmentCount - 2].filter((index) => index >= 0))
    );
  }
  return Array.from(
    { length: lastEmbeddableSegmentIndex(duration) + 1 },
    (_, index) => lastEmbeddableSegmentIndex(duration) - index
  );
}

export function segmentEndSeconds(segmentIndex: number, duration: number): number {
  return Math.min(duration, segmentIndex * SEGMENT_HOP_SECONDS + SEGMENT_WINDOW_SECONDS);
}

export function segmentTriggerSeconds(segmentIndex: number, duration: number): number {
  return Math.max(0, segmentEndSeconds(segmentIndex, duration) - CROSSFADE_SECONDS);
}

function lastEmbeddableSegmentIndex(duration: number): number {
  return Math.max(0, Math.floor(Math.max(0, duration - SEGMENT_WINDOW_SECONDS) / SEGMENT_HOP_SECONDS));
}

function segmentKey(trackId: string, segmentIndex: number): string {
  return `${trackId}:${segmentIndex}`;
}
