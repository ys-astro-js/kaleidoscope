import { type Dispatch, type SetStateAction, useEffect, useState } from "react";
import { fetchSimilarSegments } from "../api";
import {
  AUTOPLAY_HISTORY_LIMIT,
  AUTOPLAY_TRIGGER_SECONDS,
  SEGMENT_HOP_SECONDS,
  buildTrackAutoplayCandidates,
  chooseTrackAutoplayCandidate,
  segmentAutoplaySourceSegmentIndex,
  segmentEndSeconds,
  segmentTriggerSeconds,
  trackAutoplaySourceSegmentIndexes,
  type RoutePreview,
  type SegmentAutoplayPreview,
  type SegmentVisit,
  type TrackAutoplayCandidate,
  type TrackAutoplayPreview,
} from "../autoplay";
import type { SimilarSegment, SimilarityMode, Track } from "../types";
import type { AudioDeck } from "../components/NowPlaying";

type MutableRef<T> = {
  current: T;
};

type UseAutoplayRoutingOptions = {
  selected: Track | null;
  selectedId: string | null;
  selectedSegmentCount: number;
  currentSegmentIndex: number;
  currentTime: number;
  currentDuration: number;
  tracks: Track[];
  tracksRef: MutableRef<Track[]>;
  routePreview: RoutePreview | null;
  setRoutePreview: Dispatch<SetStateAction<RoutePreview | null>>;
  trackAutoplayPreview: TrackAutoplayPreview | null;
  setTrackAutoplayPreview: Dispatch<SetStateAction<TrackAutoplayPreview | null>>;
  trackAutoplayTarget: Track | null;
  similarityMode: SimilarityMode;
  segmentAutoplay: boolean;
  trackAutoplay: boolean;
  isCrossfadingRef: MutableRef<boolean>;
  autoplayKeyRef: MutableRef<string>;
  trackAutoplayKeyRef: MutableRef<string>;
  trackAutoplayHistoryRef: MutableRef<string[]>;
  trackAutoplayPreviewKeyRef: MutableRef<string>;
  activeDeckRef: MutableRef<AudioDeck>;
  getAudio: (deck: AudioDeck) => HTMLAudioElement | null;
  recordAutoplayVisit: (trackId: string, segmentIndex: number) => void;
  recordTrackAutoplayVisit: (trackId: string) => void;
  chooseAutoplayMatchFromHistory: (
    matches: SimilarSegment[],
    allTracks: Track[],
    currentVisit?: SegmentVisit
  ) => { match: SimilarSegment; track: Track } | null;
  buildRoutePreviewFromHistory: (
    source: SegmentVisit,
    matches: SimilarSegment[],
    allTracks: Track[],
    includeNext: boolean
  ) => RoutePreview;
  crossfadeToTrack: (
    track: Track,
    startSeconds: number,
    options?: { onCommit?: () => void; updateSegmentRoute?: boolean }
  ) => void;
};

export function useAutoplayRouting({
  selected,
  selectedId,
  selectedSegmentCount,
  currentSegmentIndex,
  currentTime,
  currentDuration,
  tracks,
  tracksRef,
  routePreview,
  setRoutePreview,
  trackAutoplayPreview,
  setTrackAutoplayPreview,
  trackAutoplayTarget,
  similarityMode,
  segmentAutoplay,
  trackAutoplay,
  isCrossfadingRef,
  autoplayKeyRef,
  trackAutoplayKeyRef,
  trackAutoplayHistoryRef,
  trackAutoplayPreviewKeyRef,
  activeDeckRef,
  getAudio,
  recordAutoplayVisit,
  recordTrackAutoplayVisit,
  chooseAutoplayMatchFromHistory,
  buildRoutePreviewFromHistory,
  crossfadeToTrack,
}: UseAutoplayRoutingOptions) {
  const [segmentAutoplayPreview, setSegmentAutoplayPreview] =
    useState<SegmentAutoplayPreview | null>(null);

  useEffect(() => {
    if (!selectedId || similarityMode !== "segment") {
      setRoutePreview(null);
      return;
    }
    if (isCrossfadingRef.current) {
      return;
    }

    const controller = new AbortController();
    const requestSource = { trackId: selectedId, segmentIndex: currentSegmentIndex };
    fetchSimilarSegments(requestSource.trackId, requestSource.segmentIndex, controller.signal)
      .then((matches) => {
        setRoutePreview(
          buildRoutePreviewFromHistory(
            requestSource,
            matches,
            tracksRef.current,
            false
          )
        );
      })
      .catch((error: Error) => {
        if (error.name !== "AbortError") {
          setRoutePreview(null);
        }
      });
    return () => controller.abort();
  }, [
    buildRoutePreviewFromHistory,
    currentSegmentIndex,
    isCrossfadingRef,
    selectedId,
    setRoutePreview,
    similarityMode,
    tracksRef,
  ]);

  useEffect(() => {
    if (!routePreview) {
      return;
    }
    const nextTrackId =
      similarityMode === "segment" &&
      segmentAutoplay &&
      segmentAutoplayPreview &&
      routePreview.matches.some((match) => match.id === segmentAutoplayPreview.nextTrackId)
        ? segmentAutoplayPreview.nextTrackId
        : null;
    if (routePreview.nextTrackId === nextTrackId) {
      return;
    }
    setRoutePreview({ ...routePreview, nextTrackId });
  }, [
    routePreview,
    segmentAutoplay,
    segmentAutoplayPreview,
    setRoutePreview,
    similarityMode,
  ]);

  useEffect(() => {
    if (!segmentAutoplay || similarityMode !== "segment" || !selectedId) {
      setSegmentAutoplayPreview(null);
      return;
    }
    if (isCrossfadingRef.current) {
      return;
    }

    const controller = new AbortController();
    const preferredSourceIndex = segmentAutoplaySourceSegmentIndex(
      currentSegmentIndex,
      selectedSegmentCount
    );

    const loadPreview = async () => {
      let source = { trackId: selectedId, segmentIndex: preferredSourceIndex };
      let matches = await fetchSimilarSegments(source.trackId, source.segmentIndex, controller.signal);
      let target = chooseAutoplayMatchFromHistory(matches, tracksRef.current, source);

      if (
        !target &&
        selectedSegmentCount <= 0 &&
        preferredSourceIndex !== currentSegmentIndex
      ) {
        source = { trackId: selectedId, segmentIndex: currentSegmentIndex };
        matches = await fetchSimilarSegments(source.trackId, source.segmentIndex, controller.signal);
        target = chooseAutoplayMatchFromHistory(matches, tracksRef.current, source);
      }

      if (!target) {
        setSegmentAutoplayPreview(null);
        return;
      }

      setSegmentAutoplayPreview({
        ...source,
        match: target.match,
        nextTrackId: target.track.id,
      });
    };

    loadPreview().catch((error: Error) => {
      if (error.name !== "AbortError") {
        setSegmentAutoplayPreview(null);
      }
    });
    return () => controller.abort();
  }, [
    chooseAutoplayMatchFromHistory,
    currentSegmentIndex,
    isCrossfadingRef,
    segmentAutoplay,
    selectedId,
    selectedSegmentCount,
    similarityMode,
    tracksRef,
  ]);

  useEffect(() => {
    if (!segmentAutoplay || similarityMode !== "segment" || !selected) {
      autoplayKeyRef.current = "";
      setSegmentAutoplayPreview(null);
      return;
    }
    if (isCrossfadingRef.current) {
      return;
    }

    const audio = getAudio(activeDeckRef.current);
    const playbackTime = audio?.currentTime ?? currentTime;
    const secondsIntoSegment = playbackTime - currentSegmentIndex * SEGMENT_HOP_SECONDS;
    if (secondsIntoSegment < AUTOPLAY_TRIGGER_SECONDS) {
      return;
    }
    if (
      !segmentAutoplayPreview ||
      segmentAutoplayPreview.trackId !== selected.id ||
      !isSegmentAutoplayPreviewForCurrentSegment(
        segmentAutoplayPreview,
        currentSegmentIndex,
        selectedSegmentCount
      )
    ) {
      return;
    }

    recordAutoplayVisit(selected.id, currentSegmentIndex);
    const match = segmentAutoplayPreview.match;
    const targetTrack = tracks.find((track) => track.id === segmentAutoplayPreview.nextTrackId);
    if (!match || !targetTrack) {
      return;
    }

    const autoplayKey = `${selected.id}:${currentSegmentIndex}:${segmentAutoplayPreview.segmentIndex}:${match.id}:${match.segment_index}`;
    if (autoplayKeyRef.current === autoplayKey) {
      return;
    }
    autoplayKeyRef.current = autoplayKey;
    crossfadeToTrack(targetTrack, match.start_seconds);
  }, [
    autoplayKeyRef,
    activeDeckRef,
    crossfadeToTrack,
    currentSegmentIndex,
    currentTime,
    getAudio,
    isCrossfadingRef,
    recordAutoplayVisit,
    segmentAutoplay,
    segmentAutoplayPreview,
    selected,
    selectedSegmentCount,
    similarityMode,
    tracks
  ]);

  useEffect(() => {
    if (!trackAutoplay || similarityMode !== "track" || !selectedId || currentDuration <= 0) {
      trackAutoplayPreviewKeyRef.current = "";
      setTrackAutoplayPreview(null);
      return;
    }

    const storedSegmentCount = selectedSegmentCount;
    const sourceSegmentIndexes = trackAutoplaySourceSegmentIndexes(storedSegmentCount, currentDuration);
    if (sourceSegmentIndexes.length === 0) {
      trackAutoplayPreviewKeyRef.current = "";
      setTrackAutoplayPreview(null);
      return;
    }
    const previewKey = `${selectedId}:${storedSegmentCount || "probe"}:${sourceSegmentIndexes.join(",")}:${trackAutoplayHistoryRef.current.join(",")}`;
    if (trackAutoplayPreviewKeyRef.current === previewKey) {
      return;
    }
    trackAutoplayPreviewKeyRef.current = previewKey;

    const controller = new AbortController();
    const loadPreview = async () => {
      const candidates: TrackAutoplayCandidate[] = [];
      let resolvedSourceCount = 0;
      for (const segmentIndex of sourceSegmentIndexes) {
        const matches = await fetchSimilarSegments(
          selectedId,
          segmentIndex,
          controller.signal,
          AUTOPLAY_HISTORY_LIMIT + 5
        );
        const sourceCandidates = buildTrackAutoplayCandidates(matches, tracksRef.current, {
          trackId: selectedId,
          segmentIndex,
          sourceEndSeconds: segmentEndSeconds(segmentIndex, currentDuration),
          triggerSeconds: segmentTriggerSeconds(segmentIndex, currentDuration),
        });
        if (sourceCandidates.length === 0) {
          continue;
        }
        candidates.push(...sourceCandidates);
        resolvedSourceCount += 1;
        if (storedSegmentCount === 0 && resolvedSourceCount === 2) {
          break;
        }
      }
      const target = chooseTrackAutoplayCandidate(candidates, trackAutoplayHistoryRef.current);
      if (target) {
        setTrackAutoplayPreview({
          trackId: selectedId,
          segmentIndex: target.segmentIndex,
          match: target.match,
          nextTrackId: target.track.id,
          sourceEndSeconds: target.sourceEndSeconds,
          triggerSeconds: target.triggerSeconds,
        });
        return;
      }
      setTrackAutoplayPreview(null);
    };

    loadPreview().catch((error: Error) => {
      if (error.name !== "AbortError") {
        trackAutoplayPreviewKeyRef.current = "";
        setTrackAutoplayPreview(null);
      }
    });
    return () => controller.abort();
  }, [
    currentDuration,
    selectedId,
    selectedSegmentCount,
    setTrackAutoplayPreview,
    similarityMode,
    trackAutoplay,
    trackAutoplayHistoryRef,
    trackAutoplayPreviewKeyRef,
    tracksRef,
  ]);

  useEffect(() => {
    if (
      !trackAutoplay ||
      similarityMode !== "track" ||
      !selected ||
      !trackAutoplayTarget ||
      !trackAutoplayPreview
    ) {
      trackAutoplayKeyRef.current = "";
      return;
    }
    if (isCrossfadingRef.current) {
      return;
    }

    const audio = getAudio(activeDeckRef.current);
    if (!audio || !Number.isFinite(audio.duration) || audio.duration <= 0) {
      return;
    }
    if (trackAutoplayPreview.trackId !== selected.id) {
      return;
    }
    if (audio.currentTime < trackAutoplayPreview.triggerSeconds) {
      return;
    }

    const match = trackAutoplayPreview.match;
    const autoplayKey = `${selected.id}:${trackAutoplayPreview.segmentIndex}:${match.id}:${match.segment_index}`;
    if (trackAutoplayKeyRef.current === autoplayKey) {
      return;
    }
    trackAutoplayKeyRef.current = autoplayKey;
    recordTrackAutoplayVisit(selected.id);
    crossfadeToTrack(trackAutoplayTarget, match.start_seconds, {
      updateSegmentRoute: false,
      onCommit: () => recordTrackAutoplayVisit(trackAutoplayTarget.id),
    });
  }, [
    activeDeckRef,
    crossfadeToTrack,
    currentTime,
    getAudio,
    isCrossfadingRef,
    recordTrackAutoplayVisit,
    selected,
    similarityMode,
    trackAutoplay,
    trackAutoplayKeyRef,
    trackAutoplayPreview,
    trackAutoplayTarget,
  ]);
}

function isSegmentAutoplayPreviewForCurrentSegment(
  preview: SegmentAutoplayPreview,
  currentSegmentIndex: number,
  selectedSegmentCount: number
): boolean {
  const preferredSourceIndex = segmentAutoplaySourceSegmentIndex(
    currentSegmentIndex,
    selectedSegmentCount
  );
  if (preview.segmentIndex === preferredSourceIndex) {
    return true;
  }
  return selectedSegmentCount <= 0 && preview.segmentIndex === currentSegmentIndex;
}
