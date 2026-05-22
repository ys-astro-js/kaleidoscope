import {
  type Dispatch,
  type SetStateAction,
  type SyntheticEvent,
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";
import { audioUrl, fetchSimilarSegments } from "../api";
import {
  CROSSFADE_SECONDS,
  segmentIndexForTime,
  type RoutePreview,
  type SegmentVisit,
  type TrackAutoplayPreview,
} from "../autoplay";
import type { AudioDeck } from "../components/NowPlaying";
import type { SimilarSegment, SimilarityMode, Track } from "../types";

type MutableRef<T> = {
  current: T;
};

type CrossfadeOptions = {
  onCommit?: () => void;
  updateSegmentRoute?: boolean;
};

type UseAudioPlaybackOptions = {
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

  const getAudio = useCallback((deck: AudioDeck): HTMLAudioElement | null =>
    deck === "primary" ? primaryAudioRef.current : secondaryAudioRef.current, []);

  const activateDeck = useCallback((deck: AudioDeck) => {
    activeDeckRef.current = deck;
    setActiveDeck(deck);
  }, []);

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
  }, [cancelFade]);

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
    toAudio.src = audioUrl(track.id);
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
    tracksRef,
  ]);

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

  return {
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
