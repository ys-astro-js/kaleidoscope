import { useCallback, useRef } from "react";
import {
  AUTOPLAY_HISTORY_LIMIT,
  buildRoutePreview,
  chooseAutoplayMatch,
  type RoutePreview,
  type SegmentVisit,
} from "../autoplay";
import type { SimilarSegment, Track } from "../types";

export function useAutoplayHistory() {
  const autoplayKeyRef = useRef("");
  const autoplayHistoryRef = useRef<SegmentVisit[]>([]);
  const trackAutoplayKeyRef = useRef("");
  const trackAutoplayHistoryRef = useRef<string[]>([]);
  const trackAutoplayPreviewKeyRef = useRef("");

  const recordTrackAutoplayVisit = useCallback((trackId: string) => {
    const history = trackAutoplayHistoryRef.current;
    if (history.at(-1) === trackId) {
      return;
    }
    trackAutoplayHistoryRef.current = [...history, trackId].slice(-AUTOPLAY_HISTORY_LIMIT);
  }, []);

  const resetTrackAutoplayHistory = useCallback((trackId?: string) => {
    trackAutoplayKeyRef.current = "";
    trackAutoplayHistoryRef.current = [];
    if (trackId) {
      recordTrackAutoplayVisit(trackId);
    }
  }, [recordTrackAutoplayVisit]);

  const recordAutoplayVisit = useCallback((trackId: string, segmentIndex: number) => {
    const visit = { trackId, segmentIndex };
    const history = autoplayHistoryRef.current;
    const last = history.at(-1);
    if (last?.trackId === visit.trackId && last.segmentIndex === visit.segmentIndex) {
      return;
    }
    autoplayHistoryRef.current = [...history, visit].slice(-AUTOPLAY_HISTORY_LIMIT);
  }, []);

  const resetAutoplayHistory = useCallback((trackId?: string, segmentIndex?: number) => {
    autoplayKeyRef.current = "";
    autoplayHistoryRef.current = [];
    if (trackId !== undefined && segmentIndex !== undefined) {
      recordAutoplayVisit(trackId, segmentIndex);
    }
  }, [recordAutoplayVisit]);

  const chooseAutoplayMatchFromHistory = useCallback((
    matches: SimilarSegment[],
    allTracks: Track[],
    currentVisit?: SegmentVisit
  ) => chooseAutoplayMatch(matches, allTracks, autoplayHistoryRef.current, currentVisit), []);

  const buildRoutePreviewFromHistory = useCallback((
    source: SegmentVisit,
    matches: SimilarSegment[],
    allTracks: Track[],
    includeNext: boolean
  ): RoutePreview => buildRoutePreview(
    source,
    matches,
    allTracks,
    autoplayHistoryRef.current,
    includeNext
  ), []);

  return {
    autoplayKeyRef,
    trackAutoplayKeyRef,
    trackAutoplayHistoryRef,
    trackAutoplayPreviewKeyRef,
    recordTrackAutoplayVisit,
    resetTrackAutoplayHistory,
    recordAutoplayVisit,
    resetAutoplayHistory,
    chooseAutoplayMatchFromHistory,
    buildRoutePreviewFromHistory,
  };
}
