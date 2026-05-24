import {
  type Dispatch,
  type SetStateAction,
  type SyntheticEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { audioUrl, fetchSimilarSegments, resolveAudioStem } from "../api";
import {
  CROSSFADE_SECONDS,
  segmentIndexForTime,
  type RoutePreview,
  type SegmentVisit,
  type TrackAutoplayPreview,
} from "../autoplay";
import type { AudioDeck } from "../components/NowPlaying";
import type { AudioStem, SimilarSegment, SimilarityMode, Track } from "../types";

type MutableRef<T> = {
  current: T;
};

type CrossfadeOptions = {
  onCommit?: () => void;
  updateSegmentRoute?: boolean;
};

type UseAudioPlaybackOptions = {
  selected: Track | null;
  playbackStem: AudioStem;
  similarityMode: SimilarityMode;
  trackAutoplay: boolean;
  segmentAutoplay: boolean;
  tracksRef: MutableRef<Track[]>;
  trackAutoplayPreviewKeyRef: MutableRef<string>;
  setSelected: Dispatch<SetStateAction<Track | null>>;
  setRoutePreview: Dispatch<SetStateAction<RoutePreview | null>>;
  setTrackAutoplayPreview: Dispatch<SetStateAction<TrackAutoplayPreview | null>>;
  resetAutoplayHistory: (trackId?: string, segmentIndex?: number) => void;
  resetTrackAutoplayHistory: (trackId?: string) => void;
  recordAutoplayVisit: (trackId: string, segmentIndex: number) => void;
  buildRoutePreviewFromHistory: (
    source: SegmentVisit,
    matches: SimilarSegment[],
    allTracks: Track[],
    includeNext: boolean
  ) => RoutePreview;
};

export function useAudioPlayback({
  selected,
  playbackStem,
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
}: UseAudioPlaybackOptions) {
  const primaryAudioRef = useRef<HTMLAudioElement | null>(null);
  const secondaryAudioRef = useRef<HTMLAudioElement | null>(null);
  const [activeDeck, setActiveDeck] = useState<AudioDeck>("primary");
  const activeDeckRef = useRef<AudioDeck>("primary");
  const fadeFrameRef = useRef<number | null>(null);
  const fadeIntervalRef = useRef<number | null>(null);
  const fadeTimeoutRef = useRef<number | null>(null);
  const isCrossfadingRef = useRef(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [currentDuration, setCurrentDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const getAudio = useCallback((deck: AudioDeck): HTMLAudioElement | null =>
    deck === "primary" ? primaryAudioRef.current : secondaryAudioRef.current, []);

  const trackAudioUrl = useCallback((track: Track): string =>
    audioUrl(track.id, resolveAudioStem(track, playbackStem)), [playbackStem]);

  const activateDeck = useCallback((deck: AudioDeck) => {
    activeDeckRef.current = deck;
    setActiveDeck(deck);
    const audio = getAudio(deck);
    setIsPlaying(Boolean(audio && !audio.paused));
  }, [getAudio]);

  const cancelFade = useCallback(() => {
    if (fadeFrameRef.current !== null) {
      window.cancelAnimationFrame(fadeFrameRef.current);
      fadeFrameRef.current = null;
    }
    if (fadeIntervalRef.current !== null) {
      window.clearInterval(fadeIntervalRef.current);
      fadeIntervalRef.current = null;
    }
    if (fadeTimeoutRef.current !== null) {
      window.clearTimeout(fadeTimeoutRef.current);
      fadeTimeoutRef.current = null;
    }
    isCrossfadingRef.current = false;
  }, []);

  const pauseAllAudio = useCallback(() => {
    cancelFade();
    pauseAudio(primaryAudioRef.current);
    pauseAudio(secondaryAudioRef.current);
    setIsPlaying(false);
  }, [cancelFade]);

  useEffect(() => {
    const syncPlaybackState = () => {
      const audio = getAudio(activeDeckRef.current);
      setIsPlaying(Boolean(audio && !audio.paused));
    };
    const primary = primaryAudioRef.current;
    const secondary = secondaryAudioRef.current;
    primary?.addEventListener("play", syncPlaybackState);
    primary?.addEventListener("pause", syncPlaybackState);
    primary?.addEventListener("ended", syncPlaybackState);
    secondary?.addEventListener("play", syncPlaybackState);
    secondary?.addEventListener("pause", syncPlaybackState);
    secondary?.addEventListener("ended", syncPlaybackState);
    syncPlaybackState();
    return () => {
      primary?.removeEventListener("play", syncPlaybackState);
      primary?.removeEventListener("pause", syncPlaybackState);
      primary?.removeEventListener("ended", syncPlaybackState);
      secondary?.removeEventListener("play", syncPlaybackState);
      secondary?.removeEventListener("pause", syncPlaybackState);
      secondary?.removeEventListener("ended", syncPlaybackState);
    };
  }, [getAudio]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      const audio = getAudio(activeDeckRef.current);
      if (!audio || audio.paused) {
        return;
      }
      setCurrentTime(audio.currentTime);
      if (Number.isFinite(audio.duration)) {
        setCurrentDuration(audio.duration);
      }
    }, 500);
    return () => window.clearInterval(interval);
  }, [getAudio]);

  const playTrack = useCallback((track: Track, startSeconds?: number) => {
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
      resetAutoplayHistory(track.id, segmentIndexForTime(startSeconds ?? 0));
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
    audio.src = trackAudioUrl(track);
    audio.onloadedmetadata = () => {
      const defaultStartSeconds = similarityMode === "track" && !trackAutoplay ? 39 : 0;
      const nextCurrentTime =
        startSeconds === undefined
          ? Math.min(defaultStartSeconds, Math.max(0, audio.duration - 1))
          : Math.min(startSeconds, Math.max(0, audio.duration - 1));
      audio.currentTime = nextCurrentTime;
      setCurrentTime(nextCurrentTime);
      setCurrentDuration(audio.duration);
      void audio
        .play()
        .then(() => setIsPlaying(true))
        .catch(() => setIsPlaying(false));
    };
  }, [
    cancelFade,
    getAudio,
    resetAutoplayHistory,
    resetTrackAutoplayHistory,
    setRoutePreview,
    setSelected,
    setTrackAutoplayPreview,
    similarityMode,
    trackAutoplay,
    trackAutoplayPreviewKeyRef,
    trackAudioUrl,
  ]);

  const crossfadeToTrack = useCallback((
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
    const nextSegmentIndex = segmentIndexForTime(startSeconds);
    let prefetchedMatches: SimilarSegment[] | null = null;
    let transitionCommitted = false;
    if (updateSegmentRoute) {
      void fetchSimilarSegments(track.id, nextSegmentIndex)
        .then((matches) => {
          prefetchedMatches = matches;
          if (transitionCommitted) {
            setRoutePreview(
              buildRoutePreviewFromHistory(
                { trackId: track.id, segmentIndex: nextSegmentIndex },
                matches,
                tracksRef.current,
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
    toAudio.src = trackAudioUrl(track);
    toAudio.onloadedmetadata = () => {
      toAudio.currentTime = Math.min(startSeconds, Math.max(0, toAudio.duration - 1));
      void toAudio
        .play()
        .then(() => {
          const fadeStart = performance.now();
          const commitTransition = () => {
            if (transitionCommitted) {
              return;
            }
            transitionCommitted = true;
            if (fadeFrameRef.current !== null) {
              window.cancelAnimationFrame(fadeFrameRef.current);
              fadeFrameRef.current = null;
            }
            if (fadeIntervalRef.current !== null) {
              window.clearInterval(fadeIntervalRef.current);
              fadeIntervalRef.current = null;
            }
            if (fadeTimeoutRef.current !== null) {
              window.clearTimeout(fadeTimeoutRef.current);
              fadeTimeoutRef.current = null;
            }

            fromAudio.pause();
            fromAudio.volume = 1;
            toAudio.volume = 1;
            isCrossfadingRef.current = false;
            options.onCommit?.();
            setTrackAutoplayPreview(null);
            trackAutoplayPreviewKeyRef.current = "";
            if (updateSegmentRoute) {
              recordAutoplayVisit(track.id, nextSegmentIndex);
            }
            if (updateSegmentRoute && prefetchedMatches) {
              setRoutePreview(
                buildRoutePreviewFromHistory(
                  { trackId: track.id, segmentIndex: nextSegmentIndex },
                  prefetchedMatches,
                  tracksRef.current,
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
          const updateFade = (now: number): boolean => {
            if (transitionCommitted) {
              return true;
            }
            const progress = Math.min(1, (now - fadeStart) / (CROSSFADE_SECONDS * 1000));
            const eased = progress * progress * (3 - 2 * progress);
            fromAudio.volume = 1 - eased;
            toAudio.volume = eased;

            if (progress >= 1) {
              commitTransition();
              return true;
            }
            return false;
          };
          const fade = (now: number) => {
            if (!updateFade(now)) {
              fadeFrameRef.current = window.requestAnimationFrame(fade);
            }
          };
          fadeIntervalRef.current = window.setInterval(
            () => updateFade(performance.now()),
            100
          );
          fadeTimeoutRef.current = window.setTimeout(
            commitTransition,
            CROSSFADE_SECONDS * 1000
          );
          fadeFrameRef.current = window.requestAnimationFrame(fade);
        })
        .catch(() => {
          isCrossfadingRef.current = false;
          playTrack(track, startSeconds);
        });
    };
  }, [
    activateDeck,
    buildRoutePreviewFromHistory,
    cancelFade,
    getAudio,
    playTrack,
    recordAutoplayVisit,
    segmentAutoplay,
    setRoutePreview,
    setSelected,
    setTrackAutoplayPreview,
    similarityMode,
    trackAutoplayPreviewKeyRef,
    trackAudioUrl,
    tracksRef,
  ]);

  useEffect(() => {
    if (!selected) {
      return;
    }
    const audio = getAudio(activeDeckRef.current);
    if (!audio || !audio.src) {
      return;
    }
    const nextSrc = new URL(trackAudioUrl(selected), window.location.href).href;
    if (audio.src === nextSrc) {
      return;
    }

    const nextCurrentTime = audio.currentTime;
    const shouldResume = !audio.paused;
    const handleLoadedMetadata = () => {
      const duration = Number.isFinite(audio.duration) ? audio.duration : nextCurrentTime;
      audio.currentTime = Math.min(nextCurrentTime, Math.max(0, duration - 1));
      setCurrentTime(audio.currentTime);
      setCurrentDuration(duration);
      if (shouldResume) {
        void audio
          .play()
          .then(() => setIsPlaying(true))
          .catch(() => setIsPlaying(false));
      }
    };
    audio.pause();
    audio.addEventListener("loadedmetadata", handleLoadedMetadata, { once: true });
    audio.src = nextSrc;
  }, [getAudio, playbackStem, selected, trackAudioUrl]);

  const handleTimeUpdate = useCallback((
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
  }, [currentDuration]);

  const togglePlayback = useCallback(() => {
    const audio = getAudio(activeDeckRef.current);
    if (!audio || !audio.src) {
      return;
    }
    if (audio.paused) {
      void audio
        .play()
        .then(() => setIsPlaying(true))
        .catch(() => setIsPlaying(false));
      return;
    }
    audio.pause();
    setIsPlaying(false);
  }, [getAudio]);

  const seekTo = useCallback((seconds: number) => {
    const audio = getAudio(activeDeckRef.current);
    if (!audio) {
      return;
    }
    const duration = Number.isFinite(audio.duration) ? audio.duration : currentDuration;
    const clamped = Math.min(Math.max(0, seconds), Math.max(0, duration));
    audio.currentTime = clamped;
    setCurrentTime(clamped);
  }, [currentDuration, getAudio]);

  return {
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
  };
}

function getInactiveDeck(deck: AudioDeck): AudioDeck {
  return deck === "primary" ? "secondary" : "primary";
}

function pauseAudio(audio: HTMLAudioElement | null): void {
  if (!audio) {
    return;
  }
  audio.pause();
  audio.volume = 1;
}
