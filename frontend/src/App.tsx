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

type PlaybackVisit = {
  trackId: string;
  startSeconds: number;
};

type FeedbackNotice = {
  id: number;
  label: FeedbackLabel;
};

export default function App() {
  const [selected, setSelected] = useState<Track | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [viewMode, setViewMode] = useState<ViewMode>("3d");
  const [similarityMode, setSimilarityMode] = useState<SimilarityMode>("track");
  const [trackAutoplay, setTrackAutoplay] = useState(false);
  const [segmentAutoplay, setSegmentAutoplay] = useState(false);
  const [routePreview, setRoutePreview] = useState<RoutePreview | null>(null);
  const [trackAutoplayPreview, setTrackAutoplayPreview] = useState<TrackAutoplayPreview | null>(null);
  const [playbackHistory, setPlaybackHistory] = useState<PlaybackVisit[]>([]);
  const [feedbackNotice, setFeedbackNotice] = useState<FeedbackNotice | null>(null);
  const tracksRef = useRef<Track[]>([]);
  const feedbackNoticeIdRef = useRef(0);
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
    isPlaying,
    setCurrentTime,
    primaryAudioRef,
    secondaryAudioRef,
    isCrossfadingRef,
    getAudio,
    pauseAllAudio,
    playTrack,
    crossfadeToTrack,
    togglePlayback,
    seekTo,
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

  const pushPlaybackHistory = useCallback(() => {
    if (!selected) {
      return;
    }
    const audio = getAudio(activeDeckRef.current);
    const startSeconds = audio?.currentTime ?? currentTime;
    setPlaybackHistory((history) => {
      const last = history.at(-1);
      if (last?.trackId === selected.id && Math.abs(last.startSeconds - startSeconds) < 1) {
        return history;
      }
      return [...history, { trackId: selected.id, startSeconds }].slice(-50);
    });
  }, [activeDeckRef, currentTime, getAudio, selected]);

  const playTrackWithHistory = useCallback((track: Track, startSeconds?: number) => {
    pushPlaybackHistory();
    playTrack(track, startSeconds);
  }, [playTrack, pushPlaybackHistory]);

  const crossfadeToTrackWithHistory = useCallback((
    track: Track,
    startSeconds: number,
    options?: { onCommit?: () => void; updateSegmentRoute?: boolean }
  ) => {
    pushPlaybackHistory();
    crossfadeToTrack(track, startSeconds, options);
  }, [crossfadeToTrack, pushPlaybackHistory]);

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
    crossfadeToTrack: crossfadeToTrackWithHistory,
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
    feedbackNoticeIdRef.current += 1;
    setFeedbackNotice({ id: feedbackNoticeIdRef.current, label });
    await loadTracks();
  }, [loadTracks, selected]);

  const clearFeedbackNotice = useCallback((id: number) => {
    setFeedbackNotice((current) => current?.id === id ? null : current);
  }, []);

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

  const playPrevious = useCallback(() => {
    const previous = playbackHistory.at(-1);
    if (!previous) {
      seekTo(0);
      return;
    }
    const track = tracks.find((candidate) => candidate.id === previous.trackId);
    setPlaybackHistory((history) => history.slice(0, -1));
    if (track) {
      playTrack(track, previous.startSeconds);
    }
  }, [playTrack, playbackHistory, seekTo, tracks]);

  const playNext = useCallback(() => {
    if (similarityMode === "track") {
      const target = similarTracks[0]?.track;
      if (target) {
        playTrackWithHistory(target);
      }
      return;
    }

    const target = segmentSimilarTracks[0];
    if (target) {
      playTrackWithHistory(target.track, target.startSeconds);
    }
  }, [playTrackWithHistory, segmentSimilarTracks, similarityMode, similarTracks]);

  const hasNext =
    similarityMode === "track" ? similarTracks.length > 0 : segmentSimilarTracks.length > 0;

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
        onViewModeChange={setViewMode}
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
        onSelect={playTrackWithHistory}
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
          onPlayTrack={playTrackWithHistory}
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
        similarityMode={similarityMode}
        isAutoplayActive={similarityMode === "track" ? trackAutoplay : segmentAutoplay}
        isPlaying={isPlaying}
        currentTime={currentTime}
        currentDuration={currentDuration}
        canPlayNext={hasNext}
        feedbackNotice={feedbackNotice}
        onTrackModeSelect={selectTrackMode}
        onSegmentModeSelect={selectSegmentMode}
        onPrevious={playPrevious}
        onPlayPause={togglePlayback}
        onNext={playNext}
        onAutoplayToggle={toggleAutoplay}
        onSeek={seekTo}
        onDelete={deleteSelectedTrack}
        onTimeUpdate={handleTimeUpdate}
        onFeedbackNoticeDone={clearFeedbackNotice}
      />
    </main>
  );
}
