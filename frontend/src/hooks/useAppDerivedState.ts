import { useMemo } from "react";
import {
  SEGMENT_HOP_SECONDS,
  SEGMENT_WINDOW_SECONDS,
  TRACK_AUTOPLAY_HIGHLIGHT_SECONDS,
  segmentIndexForTime,
  segmentWindowIndexForTime,
  type RoutePreview,
  type TrackAutoplayPreview,
} from "../autoplay";
import type { SimilarTrackEntry } from "../components/TrackFeedbackPanel";
import type { SegmentTrackEntry } from "../components/SegmentFeedbackPanel";
import type { SimilarityMode, Track } from "../types";

type UseAppDerivedStateOptions = {
  tracks: Track[];
  selected: Track | null;
  currentTime: number;
  routePreview: RoutePreview | null;
  trackAutoplayPreview: TrackAutoplayPreview | null;
  similarityMode: SimilarityMode;
  trackAutoplay: boolean;
  segmentAutoplay: boolean;
};

export function useAppDerivedState({
  tracks,
  selected,
  currentTime,
  routePreview,
  trackAutoplayPreview,
  similarityMode,
  trackAutoplay,
  segmentAutoplay,
}: UseAppDerivedStateOptions) {
  const selectedId = selected?.id ?? null;
  const selectedSegmentCount = selected?.segment_count ?? 0;
  const currentSegmentIndex = useMemo(
    () => {
      const index =
        similarityMode === "segment"
          ? segmentWindowIndexForTime(currentTime)
          : segmentIndexForTime(currentTime);
      if (
        similarityMode === "segment" &&
        routePreview?.trackId === selectedId &&
        currentTime < routePreview.segmentIndex * SEGMENT_HOP_SECONDS + SEGMENT_WINDOW_SECONDS
      ) {
        return routePreview.segmentIndex;
      }
      if (selectedSegmentCount <= 0) {
        return index;
      }
      return Math.min(index, selectedSegmentCount - 1);
    },
    [currentTime, routePreview, selectedId, selectedSegmentCount, similarityMode]
  );

  const similarTracks = useMemo<SimilarTrackEntry[]>(() => {
    if (!selected) {
      return [];
    }
    return selected.similar
      .map((similar) => ({
        score: similar.score,
        track: tracks.find((track) => track.id === similar.id)
      }))
      .filter((similar): similar is SimilarTrackEntry => Boolean(similar.track));
  }, [selected, tracks]);

  const trackAutoplayTarget = useMemo(() => {
    if (similarityMode !== "track" || !trackAutoplay || !trackAutoplayPreview) {
      return null;
    }
    return tracks.find((track) => track.id === trackAutoplayPreview.nextTrackId) ?? null;
  }, [similarityMode, trackAutoplay, trackAutoplayPreview, tracks]);

  const trackSceneLinks = useMemo(() => {
    if (similarityMode !== "track" || !trackAutoplay || !selected || !trackAutoplayPreview) {
      return null;
    }
    return [
      {
        id: trackAutoplayPreview.match.id,
        score: trackAutoplayPreview.match.score
      },
      ...selected.similar.filter((similar) => similar.id !== trackAutoplayPreview.match.id),
    ];
  }, [selected, similarityMode, trackAutoplay, trackAutoplayPreview]);

  const shouldHighlightTrackAutoplayTarget =
    similarityMode === "track" &&
    trackAutoplay &&
    trackAutoplayPreview !== null &&
    trackAutoplayPreview.triggerSeconds - currentTime <= TRACK_AUTOPLAY_HIGHLIGHT_SECONDS;

  const displaySegmentMatches = routePreview?.matches ?? [];
  const displaySegmentIndex = routePreview?.segmentIndex ?? currentSegmentIndex;
  const segmentSimilarTracks = useMemo<SegmentTrackEntry[]>(
    () =>
      displaySegmentMatches
        .map((similar) => ({
          score: similar.score,
          segmentIndex: similar.segment_index,
          startSeconds: similar.start_seconds,
          track: tracks.find((track) => track.id === similar.id)
        }))
        .filter((similar): similar is SegmentTrackEntry => Boolean(similar.track)),
    [displaySegmentMatches, tracks]
  );

  const focusedTrackIds = useMemo(() => {
    if (similarityMode === "track" && trackAutoplay && selectedId) {
      return [
        selectedId,
        ...similarTracks.map(({ track }) => track.id),
        ...(trackAutoplayTarget ? [trackAutoplayTarget.id] : []),
      ];
    }
    if (similarityMode !== "segment" || !segmentAutoplay || !routePreview) {
      return null;
    }
    return [
      routePreview.trackId,
      ...routePreview.matches.map((match) => match.id),
      ...(routePreview.nextTrackId ? [routePreview.nextTrackId] : []),
    ];
  }, [
    routePreview,
    segmentAutoplay,
    selectedId,
    similarityMode,
    similarTracks,
    trackAutoplay,
    trackAutoplayTarget,
  ]);

  return {
    selectedId,
    selectedSegmentCount,
    currentSegmentIndex,
    similarTracks,
    trackAutoplayTarget,
    trackSceneLinks,
    shouldHighlightTrackAutoplayTarget,
    displaySegmentIndex,
    segmentSimilarTracks,
    focusedTrackIds,
  };
}
