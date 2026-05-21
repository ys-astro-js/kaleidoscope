import { type SyntheticEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  audioUrl,
  deleteAllTracks,
  deleteTrack,
  fetchSimilarSegments,
  fetchTracks,
  submitFeedback,
  uploadTracks
} from "./api";
import MusicScene from "./MusicScene";
import type { FeedbackLabel, SimilarSegment, SimilarityMode, Track, ViewMode } from "./types";

const SEGMENT_HOP_SECONDS = 15;
const SEGMENT_WINDOW_SECONDS = 30;
const CROSSFADE_SECONDS = 4;
const TRACK_AUTOPLAY_HIGHLIGHT_SECONDS = 5;
const AUTOPLAY_TRIGGER_SECONDS = SEGMENT_HOP_SECONDS - CROSSFADE_SECONDS;
const AUTOPLAY_HISTORY_LIMIT = 12;
type AudioDeck = "primary" | "secondary";
type SegmentVisit = {
  trackId: string;
  segmentIndex: number;
};
type RoutePreview = SegmentVisit & {
  matches: SimilarSegment[];
  nextTrackId: string | null;
};
type TrackAutoplayPreview = SegmentVisit & {
  match: SimilarSegment;
  nextTrackId: string;
  sourceEndSeconds: number;
  triggerSeconds: number;
};
type TrackAutoplayCandidate = SegmentVisit & {
  match: SimilarSegment;
  track: Track;
  sourceEndSeconds: number;
  triggerSeconds: number;
};
type CrossfadeOptions = {
  onCommit?: () => void;
  updateSegmentRoute?: boolean;
};

export default function App() {
  const [tracks, setTracks] = useState<Track[]>([]);
  const [selected, setSelected] = useState<Track | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [message, setMessage] = useState("");
  const [viewMode, setViewMode] = useState<ViewMode>("3d");
  const [similarityMode, setSimilarityMode] = useState<SimilarityMode>("track");
  const [trackAutoplay, setTrackAutoplay] = useState(false);
  const [segmentAutoplay, setSegmentAutoplay] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [currentDuration, setCurrentDuration] = useState(0);
  const [routePreview, setRoutePreview] = useState<RoutePreview | null>(null);
  const [trackAutoplayPreview, setTrackAutoplayPreview] = useState<TrackAutoplayPreview | null>(null);
  const [dismissedErrorIds, setDismissedErrorIds] = useState<Set<string>>(() => new Set());
  const primaryAudioRef = useRef<HTMLAudioElement | null>(null);
  const secondaryAudioRef = useRef<HTMLAudioElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const tracksRef = useRef<Track[]>([]);
  const [activeDeck, setActiveDeck] = useState<AudioDeck>("primary");
  const activeDeckRef = useRef<AudioDeck>("primary");
  const fadeFrameRef = useRef<number | null>(null);
  const isCrossfadingRef = useRef(false);
  const autoplayKeyRef = useRef("");
  const autoplayHistoryRef = useRef<SegmentVisit[]>([]);
  const trackAutoplayKeyRef = useRef("");
  const trackAutoplayHistoryRef = useRef<string[]>([]);
  const trackAutoplayPreviewKeyRef = useRef("");
  const previousSimilarityModeRef = useRef<SimilarityMode>(similarityMode);

  const getAudio = (deck: AudioDeck): HTMLAudioElement | null =>
    deck === "primary" ? primaryAudioRef.current : secondaryAudioRef.current;

  const getInactiveDeck = (deck: AudioDeck): AudioDeck =>
    deck === "primary" ? "secondary" : "primary";

  const activateDeck = (deck: AudioDeck) => {
    activeDeckRef.current = deck;
    setActiveDeck(deck);
  };

  const cancelFade = () => {
    if (fadeFrameRef.current !== null) {
      window.cancelAnimationFrame(fadeFrameRef.current);
      fadeFrameRef.current = null;
    }
    isCrossfadingRef.current = false;
  };

  const pauseAudio = (audio: HTMLAudioElement | null) => {
    if (!audio) {
      return;
    }
    audio.pause();
    audio.volume = 1;
  };

  const pauseAllAudio = () => {
    cancelFade();
    pauseAudio(primaryAudioRef.current);
    pauseAudio(secondaryAudioRef.current);
  };

  const recordTrackAutoplayVisit = (trackId: string) => {
    const history = trackAutoplayHistoryRef.current;
    if (history.at(-1) === trackId) {
      return;
    }
    trackAutoplayHistoryRef.current = [...history, trackId].slice(-AUTOPLAY_HISTORY_LIMIT);
  };

  const resetTrackAutoplayHistory = (trackId?: string) => {
    trackAutoplayKeyRef.current = "";
    trackAutoplayHistoryRef.current = [];
    if (trackId) {
      recordTrackAutoplayVisit(trackId);
    }
  };

  const recordAutoplayVisit = (trackId: string, segmentIndex: number) => {
    const visit = { trackId, segmentIndex };
    const history = autoplayHistoryRef.current;
    const last = history.at(-1);
    if (last?.trackId === visit.trackId && last.segmentIndex === visit.segmentIndex) {
      return;
    }
    autoplayHistoryRef.current = [...history, visit].slice(-AUTOPLAY_HISTORY_LIMIT);
  };

  const resetAutoplayHistory = (trackId?: string, segmentIndex?: number) => {
    autoplayKeyRef.current = "";
    autoplayHistoryRef.current = [];
    if (trackId !== undefined && segmentIndex !== undefined) {
      recordAutoplayVisit(trackId, segmentIndex);
    }
  };

  const segmentKey = (trackId: string, segmentIndex: number): string =>
    `${trackId}:${segmentIndex}`;

  const withHistoryVisit = (history: SegmentVisit[], visit: SegmentVisit): SegmentVisit[] => {
    const last = history.at(-1);
    if (last?.trackId === visit.trackId && last.segmentIndex === visit.segmentIndex) {
      return history;
    }
    return [...history, visit].slice(-AUTOPLAY_HISTORY_LIMIT);
  };

  const chooseAutoplayMatch = (
    matches: SimilarSegment[],
    allTracks: Track[],
    currentVisit?: SegmentVisit
  ): { match: SimilarSegment; track: Track } | null => {
    const history = currentVisit
      ? withHistoryVisit(autoplayHistoryRef.current, currentVisit)
      : autoplayHistoryRef.current;
    const recentSegmentKeys = new Set(
      history.slice(-AUTOPLAY_HISTORY_LIMIT).map((visit) => segmentKey(visit.trackId, visit.segmentIndex))
    );
    const previousVisit = history.at(-2);

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
  };

  const buildTrackAutoplayCandidates = (
    matches: SimilarSegment[],
    allTracks: Track[],
    source: SegmentVisit & { sourceEndSeconds: number; triggerSeconds: number }
  ): TrackAutoplayCandidate[] =>
    matches
      .map((match) => ({
        ...source,
        match,
        track: allTracks.find((track) => track.id === match.id)
      }))
      .filter(
        (candidate): candidate is TrackAutoplayCandidate =>
          Boolean(candidate.track)
      );

  const chooseTrackAutoplayCandidate = (
    candidates: TrackAutoplayCandidate[]
  ): TrackAutoplayCandidate | null => {
    if (candidates.length === 0) {
      return null;
    }
    const recentTrackIds = new Set(trackAutoplayHistoryRef.current.slice(-AUTOPLAY_HISTORY_LIMIT));
    const preferredCandidates = candidates.filter(({ track }) => !recentTrackIds.has(track.id));
    const pool = preferredCandidates.length > 0 ? preferredCandidates : candidates;
    return pool.reduce((best, candidate) =>
      candidate.match.score > best.match.score ? candidate : best
    );
  };

  const buildRoutePreview = (
    source: SegmentVisit,
    matches: SimilarSegment[],
    allTracks: Track[],
    includeNext: boolean
  ): RoutePreview => ({
    ...source,
    matches,
    nextTrackId: includeNext
      ? chooseAutoplayMatch(matches, allTracks, source)?.track.id ?? null
      : null,
  });

  const lastEmbeddableSegmentIndex = (duration: number): number =>
    Math.max(0, Math.floor(Math.max(0, duration - SEGMENT_WINDOW_SECONDS) / SEGMENT_HOP_SECONDS));

  const segmentEndSeconds = (segmentIndex: number, duration: number): number =>
    Math.min(duration, segmentIndex * SEGMENT_HOP_SECONDS + SEGMENT_WINDOW_SECONDS);

  const segmentTriggerSeconds = (segmentIndex: number, duration: number): number =>
    Math.max(0, segmentEndSeconds(segmentIndex, duration) - CROSSFADE_SECONDS);

  const loadTracks = useCallback(async () => {
    const nextTracks = await fetchTracks();
    setTracks(nextTracks);
    setSelected((current) => {
      if (!current) {
        return null;
      }
      return nextTracks.find((track) => track.id === current.id) ?? null;
    });
  }, []);

  useEffect(() => {
    tracksRef.current = tracks;
  }, [tracks]);

  useEffect(() => {
    loadTracks().catch(() => setMessage("Connection failed"));
    const interval = window.setInterval(() => {
      loadTracks().catch(() => setMessage("Connection failed"));
    }, 2500);
    return () => window.clearInterval(interval);
  }, [loadTracks]);

  const activeTracks = useMemo(
    () => tracks.filter((track) => track.status === "queued" || track.status === "processing"),
    [tracks]
  );
  const selectedId = selected?.id ?? null;
  const selectedSegmentCount = selected?.segment_count ?? 0;
  const failedTracks = useMemo(() => tracks.filter((track) => track.status === "error"), [tracks]);
  const visibleFailedTracks = useMemo(
    () => failedTracks.filter((track) => !dismissedErrorIds.has(track.id)),
    [dismissedErrorIds, failedTracks]
  );
  const similarTracks = useMemo(() => {
    if (!selected) {
      return [];
    }
    return selected.similar
      .map((similar) => ({
        score: similar.score,
        track: tracks.find((track) => track.id === similar.id)
      }))
      .filter((similar): similar is { score: number; track: Track } => Boolean(similar.track));
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
  const currentSegmentIndex = useMemo(
    () => Math.max(0, Math.floor(currentTime / SEGMENT_HOP_SECONDS)),
    [currentTime]
  );
  const displaySegmentMatches = routePreview?.matches ?? [];
  const displaySegmentIndex = routePreview?.segmentIndex ?? currentSegmentIndex;
  const segmentSimilarTracks = useMemo(
    () =>
      displaySegmentMatches
        .map((similar) => ({
          score: similar.score,
          segmentIndex: similar.segment_index,
          startSeconds: similar.start_seconds,
          track: tracks.find((track) => track.id === similar.id)
        }))
        .filter(
          (
            similar
          ): similar is {
            score: number;
            segmentIndex: number;
            startSeconds: number;
            track: Track;
          } => Boolean(similar.track)
        ),
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
        setRoutePreview(buildRoutePreview(requestSource, matches, tracks, segmentAutoplay));
      })
      .catch((error: Error) => {
        if (error.name !== "AbortError" && !routePreview) {
          setRoutePreview(null);
        }
      });
    return () => controller.abort();
  }, [currentSegmentIndex, selectedId, similarityMode]);

  useEffect(() => {
    if (!routePreview) {
      return;
    }
    const nextTrackId =
      similarityMode === "segment" && segmentAutoplay
        ? chooseAutoplayMatch(routePreview.matches, tracks, {
            trackId: routePreview.trackId,
            segmentIndex: routePreview.segmentIndex,
          })?.track.id ?? null
        : null;
    if (routePreview.nextTrackId === nextTrackId) {
      return;
    }
    setRoutePreview({ ...routePreview, nextTrackId });
  }, [routePreview, segmentAutoplay, similarityMode, tracks]);

  useEffect(() => {
    if (!segmentAutoplay || similarityMode !== "segment" || !selected) {
      autoplayKeyRef.current = "";
      return;
    }
    if (isCrossfadingRef.current) {
      return;
    }

    const secondsIntoSegment = currentTime - currentSegmentIndex * SEGMENT_HOP_SECONDS;
    if (secondsIntoSegment < AUTOPLAY_TRIGGER_SECONDS) {
      return;
    }
    if (
      !routePreview ||
      routePreview.trackId !== selected.id ||
      routePreview.segmentIndex !== currentSegmentIndex ||
      !routePreview.nextTrackId
    ) {
      return;
    }

    recordAutoplayVisit(selected.id, currentSegmentIndex);
    const match = routePreview.matches.find((candidate) => candidate.id === routePreview.nextTrackId);
    const targetTrack = tracks.find((track) => track.id === routePreview.nextTrackId);
    if (!match || !targetTrack) {
      return;
    }

    const autoplayKey = `${selected.id}:${currentSegmentIndex}:${match.id}:${match.segment_index}`;
    if (autoplayKeyRef.current === autoplayKey) {
      return;
    }
    autoplayKeyRef.current = autoplayKey;
    crossfadeToTrack(targetTrack, match.start_seconds);
  }, [
    currentSegmentIndex,
    currentTime,
    segmentAutoplay,
    routePreview,
    selected,
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
    const sourceSegmentIndexes =
      storedSegmentCount > 0
        ? Array.from(
            new Set([storedSegmentCount - 1, storedSegmentCount - 2].filter((index) => index >= 0))
          )
        : Array.from(
            { length: lastEmbeddableSegmentIndex(currentDuration) + 1 },
            (_, index) => lastEmbeddableSegmentIndex(currentDuration) - index
          );
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
      const target = chooseTrackAutoplayCandidate(candidates);
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
    similarityMode,
    trackAutoplay,
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
    if (currentTime < trackAutoplayPreview.triggerSeconds) {
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
    currentTime,
    selected,
    similarityMode,
    trackAutoplay,
    trackAutoplayPreview,
    trackAutoplayTarget,
  ]);

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
  }, [selected, similarityMode]);

  const handleFiles = async (files: FileList | File[]) => {
    const selectedFiles = Array.from(files).filter(isAudioFile);
    if (selectedFiles.length === 0) {
      setMessage("No audio files found");
      return;
    }
    setMessage("Uploading");
    await uploadTracks(selectedFiles);
    setMessage("");
    await loadTracks();
  };

  const playTrack = (track: Track, startSeconds?: number) => {
    cancelFade();
    const deck = activeDeckRef.current;
    const audio = getAudio(deck);
    const inactiveAudio = getAudio(getInactiveDeck(deck));

    setRoutePreview(null);
    setTrackAutoplayPreview(null);
    trackAutoplayPreviewKeyRef.current = "";
    setSelected(track);
    setCurrentTime(startSeconds ?? 0);
    setCurrentDuration(0);
    if (similarityMode === "segment") {
      resetAutoplayHistory(track.id, Math.max(0, Math.floor((startSeconds ?? 0) / SEGMENT_HOP_SECONDS)));
      resetTrackAutoplayHistory();
    } else {
      resetAutoplayHistory();
      resetTrackAutoplayHistory(track.id);
    }
    if (!audio) {
      return;
    }
    pauseAudio(inactiveAudio);
    audio.volume = 1;
    audio.src = audioUrl(track.id);
    audio.onloadedmetadata = () => {
      const defaultStartSeconds = similarityMode === "track" && !trackAutoplay ? 39 : 0;
      const nextCurrentTime =
        startSeconds === undefined
          ? Math.min(defaultStartSeconds, Math.max(0, audio.duration - 1))
          : Math.min(startSeconds, Math.max(0, audio.duration - 1));
      audio.currentTime = nextCurrentTime;
      setCurrentTime(nextCurrentTime);
      setCurrentDuration(audio.duration);
      void audio.play();
    };
  };

  const crossfadeToTrack = (
    track: Track,
    startSeconds: number,
    options: CrossfadeOptions = {}
  ) => {
    const fromDeck = activeDeckRef.current;
    const toDeck = getInactiveDeck(fromDeck);
    const fromAudio = getAudio(fromDeck);
    const toAudio = getAudio(toDeck);
    if (!fromAudio || !toAudio) {
      playTrack(track, startSeconds);
      return;
    }

    cancelFade();
    isCrossfadingRef.current = true;
    const updateSegmentRoute = options.updateSegmentRoute ?? similarityMode === "segment";
    const nextSegmentIndex = Math.max(0, Math.floor(startSeconds / SEGMENT_HOP_SECONDS));
    let prefetchedMatches: SimilarSegment[] | null = null;
    let transitionCommitted = false;
    if (updateSegmentRoute) {
      void fetchSimilarSegments(track.id, nextSegmentIndex)
        .then((matches) => {
          prefetchedMatches = matches;
          if (transitionCommitted) {
            setRoutePreview(
              buildRoutePreview(
                { trackId: track.id, segmentIndex: nextSegmentIndex },
                matches,
                tracks,
                segmentAutoplay
              )
            );
          }
        })
        .catch(() => {
          prefetchedMatches = null;
        });
    }
    toAudio.pause();
    toAudio.volume = 0;
    toAudio.src = audioUrl(track.id);
    toAudio.onloadedmetadata = () => {
      toAudio.currentTime = Math.min(startSeconds, Math.max(0, toAudio.duration - 1));
      void toAudio
        .play()
        .then(() => {
          const fadeStart = performance.now();
          const fade = (now: number) => {
            const progress = Math.min(1, (now - fadeStart) / (CROSSFADE_SECONDS * 1000));
            const eased = progress * progress * (3 - 2 * progress);
            fromAudio.volume = 1 - eased;
            toAudio.volume = eased;

            if (progress < 1) {
              fadeFrameRef.current = window.requestAnimationFrame(fade);
              return;
            }

            fromAudio.pause();
            fromAudio.volume = 1;
            toAudio.volume = 1;
            fadeFrameRef.current = null;
            isCrossfadingRef.current = false;
            transitionCommitted = true;
            options.onCommit?.();
            setTrackAutoplayPreview(null);
            trackAutoplayPreviewKeyRef.current = "";
            if (updateSegmentRoute) {
              recordAutoplayVisit(track.id, nextSegmentIndex);
            }
            if (updateSegmentRoute && prefetchedMatches) {
              setRoutePreview(
                buildRoutePreview(
                  { trackId: track.id, segmentIndex: nextSegmentIndex },
                  prefetchedMatches,
                  tracks,
                  segmentAutoplay
                )
              );
            } else if (!updateSegmentRoute) {
              setRoutePreview(null);
            }
            setSelected(track);
            setCurrentTime(toAudio.currentTime);
            setCurrentDuration(toAudio.duration);
            activateDeck(toDeck);
          };
          fadeFrameRef.current = window.requestAnimationFrame(fade);
        })
        .catch(() => {
          isCrossfadingRef.current = false;
          playTrack(track, startSeconds);
        });
    };
  };

  const handleTimeUpdate = (
    deck: AudioDeck,
    event: SyntheticEvent<HTMLAudioElement>
  ) => {
    if (activeDeckRef.current !== deck) {
      return;
    }
    setCurrentTime(event.currentTarget.currentTime);
    if (
      Number.isFinite(event.currentTarget.duration) &&
      event.currentTarget.duration !== currentDuration
    ) {
      setCurrentDuration(event.currentTarget.duration);
    }
  };

  const sendFeedback = async (candidate: Track, label: FeedbackLabel) => {
    if (!selected) {
      return;
    }
    await submitFeedback(selected.id, candidate.id, label);
    setMessage("Feedback saved");
    await loadTracks();
  };

  const removeTrack = async (track: Track) => {
    await deleteTrack(track.id);
    if (selected?.id === track.id) {
      pauseAllAudio();
      setSelected(null);
    }
    setDismissedErrorIds((current) => {
      const next = new Set(current);
      next.delete(track.id);
      return next;
    });
    await loadTracks();
  };

  const removeAllTracks = async () => {
    const confirmed = window.confirm("Delete all tracks?");
    if (!confirmed) {
      return;
    }
    await deleteAllTracks();
    pauseAllAudio();
    setSelected(null);
    setDismissedErrorIds(new Set());
    await loadTracks();
  };

  return (
    <main
      className={isDragging ? "app dragging" : "app"}
      onDragOver={(event) => {
        event.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setIsDragging(false);
        void handleFiles(event.dataTransfer.files);
      }}
    >
      <section className="topbar">
        <button type="button" onClick={() => fileInputRef.current?.click()}>
          Upload
        </button>
        {tracks.length > 0 ? (
          <button type="button" onClick={() => void removeAllTracks()}>
            Clear
          </button>
        ) : null}
        <div className="mode-toggle" role="group" aria-label="view mode">
          <button
            type="button"
            className={viewMode === "2d" ? "active" : ""}
            onClick={() => setViewMode("2d")}
          >
            2D
          </button>
          <button
            type="button"
            className={viewMode === "3d" ? "active" : ""}
            onClick={() => setViewMode("3d")}
          >
            3D
          </button>
        </div>
        <div className="mode-toggle" role="group" aria-label="similarity mode">
          <button
            type="button"
            className={similarityMode === "track" ? "active" : ""}
            onClick={() => {
              setSimilarityMode("track");
              resetAutoplayHistory();
            }}
          >
            Track
          </button>
          <button
            type="button"
            className={similarityMode === "segment" ? "active" : ""}
            onClick={() => setSimilarityMode("segment")}
          >
            Segment
          </button>
        </div>
        <button
          type="button"
          className={
            (similarityMode === "track" ? trackAutoplay : segmentAutoplay)
              ? "autoplay-toggle active"
              : "autoplay-toggle"
          }
          onClick={() => {
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
          }}
        >
          Auto
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*"
          multiple
          onChange={(event) => {
            if (event.target.files) {
              void handleFiles(event.target.files);
              event.target.value = "";
            }
          }}
        />
        {activeTracks.map((track) => (
          <span className="skeleton skeleton-chip" key={track.id} aria-label={`${track.title} processing`} />
        ))}
        {message ? <span className="status-pill">{message}</span> : null}
      </section>

      <MusicScene
        tracks={tracks}
        selectedId={selected?.id ?? null}
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

      {selected && similarityMode === "track" && similarTracks.length > 0 ? (
        <section className="feedback-panel" aria-label="similarity feedback">
          <h2>Similar tracks</h2>
          {similarTracks.map(({ score, track }) => (
            <div className="feedback-line" key={track.id}>
              <div>
                <strong>{track.title}</strong>
                <span>{Math.round(score * 100)}%</span>
              </div>
              <div className="feedback-actions">
                <button type="button" onClick={() => void sendFeedback(track, "similar")}>
                  Similar
                </button>
                <button type="button" onClick={() => void sendFeedback(track, "not_similar")}>
                  Not similar
                </button>
              </div>
            </div>
          ))}
        </section>
      ) : null}

      {selected && similarityMode === "segment" && segmentSimilarTracks.length > 0 ? (
        <section className="feedback-panel segment-panel" aria-label="similar segments">
          <h2>Segment {displaySegmentIndex + 1}</h2>
          {segmentSimilarTracks.map(({ score, segmentIndex, startSeconds, track }) => (
            <button
              type="button"
              className="segment-line"
              key={`${track.id}:${segmentIndex}`}
              onClick={() => playTrack(track, startSeconds)}
            >
              <span>
                <strong>{track.title}</strong>
                <small>Segment {segmentIndex + 1}</small>
              </span>
              <span>{Math.round(score * 100)}%</span>
            </button>
          ))}
        </section>
      ) : null}

      {activeTracks.length > 0 ? (
        <section className="skeleton-stage" aria-label="processing tracks">
          {activeTracks.map((track) => (
            <div className="skeleton skeleton-art" key={track.id} />
          ))}
        </section>
      ) : null}

      {visibleFailedTracks.length > 0 ? (
        <section className="error-panel" aria-label="track errors">
          {visibleFailedTracks.map((track) => (
            <div className="error-line" key={track.id}>
              <div>
                <strong>{track.title}</strong>
                <span>{track.error ?? "Unknown error"}</span>
              </div>
              <div className="error-actions">
                <button
                  type="button"
                  onClick={() =>
                    setDismissedErrorIds((current) => {
                      const next = new Set(current);
                      next.add(track.id);
                      return next;
                    })
                  }
                >
                  Dismiss
                </button>
                <button type="button" onClick={() => void removeTrack(track)}>
                  Delete
                </button>
              </div>
            </div>
          ))}
        </section>
      ) : null}

      <section className="now-playing">
        <div>
          <p>{selected?.title ?? ""}</p>
          <small>{selected?.artist ?? selected?.filename ?? ""}</small>
        </div>
        {selected ? (
          <button type="button" onClick={() => void removeTrack(selected)}>
            Delete
          </button>
        ) : null}
        <audio
          ref={primaryAudioRef}
          className={activeDeck === "primary" ? "active" : ""}
          controls={activeDeck === "primary"}
          onTimeUpdate={(event) => handleTimeUpdate("primary", event)}
        />
        <audio
          ref={secondaryAudioRef}
          className={activeDeck === "secondary" ? "active" : ""}
          controls={activeDeck === "secondary"}
          onTimeUpdate={(event) => handleTimeUpdate("secondary", event)}
        />
      </section>
    </main>
  );
}

function isAudioFile(file: File): boolean {
  if (file.type.startsWith("audio/")) {
    return true;
  }
  return /\.(aac|flac|m4a|mp3|ogg|wav)$/i.test(file.name);
}
