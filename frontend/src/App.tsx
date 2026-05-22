import { type DragEvent, useCallback, useEffect, useRef, useState } from "react";
import { submitFeedback } from "./api";
import {
  type RoutePreview,
  type TrackAutoplayPreview,
} from "./autoplay";
import { ErrorPanel } from "./components/ErrorPanel";
import { NowPlaying } from "./components/NowPlaying";
import { ProcessingStage } from "./components/ProcessingStage";
import { SegmentFeedbackPanel } from "./components/SegmentFeedbackPanel";
import { Topbar } from "./components/Topbar";
import { TrackFeedbackPanel } from "./components/TrackFeedbackPanel";
import { useAudioPlayback } from "./hooks/useAudioPlayback";
import { useAppDerivedState } from "./hooks/useAppDerivedState";
import { useAutoplayHistory } from "./hooks/useAutoplayHistory";
import { useAutoplayRouting } from "./hooks/useAutoplayRouting";
import { useTrackLibrary } from "./hooks/useTrackLibrary";
import MusicScene from "./MusicScene";
import type { FeedbackLabel, SimilarityMode, Track, ViewMode } from "./types";

export default function App() {
  const [selected, setSelected] = useState<Track | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>("3d");
  const [similarityMode, setSimilarityMode] = useState<SimilarityMode>("track");
  const [trackAutoplay, setTrackAutoplay] = useState(false);
  const [segmentAutoplay, setSegmentAutoplay] = useState(false);
  const [routePreview, setRoutePreview] = useState<RoutePreview | null>(null);
  const [trackAutoplayPreview, setTrackAutoplayPreview] = useState<TrackAutoplayPreview | null>(null);
  const tracksRef = useRef<Track[]>([]);
  const previousSimilarityModeRef = useRef<SimilarityMode>(similarityMode);

  const syncSelectedTrack = useCallback((nextTracks: Track[]) => {
    setSelected((current) => {
      if (!current) {
        return null;
      }
      return nextTracks.find((track) => track.id === current.id) ?? null;
    });
  }, []);

  const {
    tracks,
    activeTracks,
    visibleFailedTracks,
    message,
    setMessage,
    loadTracks,
    handleFiles,
    deleteTrackFromLibrary,
    deleteAllTracksFromLibrary,
    dismissTrackError,
  } = useTrackLibrary({ onTracksLoaded: syncSelectedTrack });

  useEffect(() => {
    tracksRef.current = tracks;
  }, [tracks]);

  const {
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
  } = useAutoplayHistory();

  const {
    activeDeck,
    activeDeckRef,
    currentTime,
    currentDuration,
    setCurrentTime,
    primaryAudioRef,
    secondaryAudioRef,
    isCrossfadingRef,
    getAudio,
    pauseAllAudio,
    playTrack,
    crossfadeToTrack,
    handleTimeUpdate,
  } = useAudioPlayback({
    similarityMode,
    trackAutoplay,
    segmentAutoplay,
    tracksRef,
    trackAutoplayPreviewKeyRef,
    setSelected,
    setRoutePreview,
    setTrackAutoplayPreview,
    resetAutoplayHistory,
    resetTrackAutoplayHistory,
    recordAutoplayVisit,
    buildRoutePreviewFromHistory,
  });

  const {
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
  } = useAppDerivedState({
    tracks,
    selected,
    currentTime,
    routePreview,
    trackAutoplayPreview,
    similarityMode,
    trackAutoplay,
    segmentAutoplay,
  });

  useAutoplayRouting({
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
  });

  useEffect(() => {
    const previousMode = previousSimilarityModeRef.current;
    previousSimilarityModeRef.current = similarityMode;
    if (previousMode === "segment" || similarityMode !== "segment" || !selected) {
      return;
    }
    const audio = getAudio(activeDeckRef.current);
    if (!audio) {
      return;
    }
    audio.currentTime = 0;
    setCurrentTime(0);
  }, [getAudio, selected, similarityMode]);

  const sendFeedback = useCallback(async (candidate: Track, label: FeedbackLabel) => {
    if (!selected) {
      return;
    }
    await submitFeedback(selected.id, candidate.id, label);
    setMessage("Feedback saved");
    await loadTracks();
  }, [loadTracks, selected, setMessage]);

  const removeTrack = useCallback(async (track: Track) => {
    await deleteTrackFromLibrary(track);
    if (selected?.id === track.id) {
      pauseAllAudio();
      setSelected(null);
    }
  }, [deleteTrackFromLibrary, pauseAllAudio, selected]);

  const removeAllTracks = useCallback(async () => {
    const confirmed = window.confirm("Delete all tracks?");
    if (!confirmed) {
      return;
    }
    await deleteAllTracksFromLibrary();
    pauseAllAudio();
    setSelected(null);
  }, [deleteAllTracksFromLibrary, pauseAllAudio]);

  const selectTrackMode = useCallback(() => {
    setSimilarityMode("track");
    resetAutoplayHistory();
  }, [resetAutoplayHistory]);

  const selectSegmentMode = useCallback(() => {
    setSimilarityMode("segment");
  }, []);

  const toggleAutoplay = useCallback(() => {
    if (similarityMode === "track") {
      const next = !trackAutoplay;
      setTrackAutoplay(next);
      resetTrackAutoplayHistory(next && selected ? selected.id : undefined);
      return;
    }

    const next = !segmentAutoplay;
    setSegmentAutoplay(next);
    if (next && selected) {
      resetAutoplayHistory(selected.id, currentSegmentIndex);
    } else {
      resetAutoplayHistory();
    }
  }, [
    currentSegmentIndex,
    resetAutoplayHistory,
    resetTrackAutoplayHistory,
    segmentAutoplay,
    selected,
    similarityMode,
    trackAutoplay,
  ]);

  const onDragOver = useCallback((event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    setIsDragging(true);
  }, []);

  const onDrop = useCallback((event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    setIsDragging(false);
    void handleFiles(event.dataTransfer.files);
  }, [handleFiles]);

  const onDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const clearTracks = useCallback(() => {
    void removeAllTracks();
  }, [removeAllTracks]);

  const selectFiles = useCallback((files: FileList) => {
    void handleFiles(files);
  }, [handleFiles]);

  const submitTrackFeedback = useCallback((track: Track, label: FeedbackLabel) => {
    void sendFeedback(track, label);
  }, [sendFeedback]);

  const deleteSelectedTrack = useCallback((track: Track) => {
    void removeTrack(track);
  }, [removeTrack]);

  return (
    <main
      className={isDragging ? "app dragging" : "app"}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <Topbar
        tracksCount={tracks.length}
        activeTracks={activeTracks}
        message={message}
        viewMode={viewMode}
        similarityMode={similarityMode}
        isAutoplayActive={similarityMode === "track" ? trackAutoplay : segmentAutoplay}
        onViewModeChange={setViewMode}
        onTrackModeSelect={selectTrackMode}
        onSegmentModeSelect={selectSegmentMode}
        onAutoplayToggle={toggleAutoplay}
        onClearTracks={clearTracks}
        onFilesSelected={selectFiles}
      />

      <MusicScene
        tracks={tracks}
        selectedId={selectedId}
        viewMode={viewMode}
        linkSourceId={similarityMode === "segment" ? routePreview?.trackId ?? null : null}
        similarLinks={similarityMode === "segment" ? routePreview?.matches ?? null : trackSceneLinks}
        highlightedLinkId={
          similarityMode === "segment" && segmentAutoplay
            ? routePreview?.nextTrackId ?? null
            : shouldHighlightTrackAutoplayTarget
              ? trackAutoplayTarget?.id ?? null
              : null
        }
        focusedTrackIds={focusedTrackIds}
        onSelect={playTrack}
      />

      {selected && similarityMode === "track" ? (
        <TrackFeedbackPanel
          entries={similarTracks}
          onFeedback={submitTrackFeedback}
        />
      ) : null}

      {selected && similarityMode === "segment" ? (
        <SegmentFeedbackPanel
          displaySegmentIndex={displaySegmentIndex}
          entries={segmentSimilarTracks}
          onPlayTrack={playTrack}
        />
      ) : null}

      <ProcessingStage tracks={activeTracks} />

      <ErrorPanel
        tracks={visibleFailedTracks}
        onDismiss={dismissTrackError}
        onDelete={deleteSelectedTrack}
      />

      <NowPlaying
        selected={selected}
        activeDeck={activeDeck}
        primaryAudioRef={primaryAudioRef}
        secondaryAudioRef={secondaryAudioRef}
        onDelete={deleteSelectedTrack}
        onTimeUpdate={handleTimeUpdate}
      />
    </main>
  );
}
