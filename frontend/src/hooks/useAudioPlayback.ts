import {
  type Dispatch,
  type SetStateAction,
  type SyntheticEvent,
  useCallback,
  useEffect,
  useMemo,
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
import type { AudioDeck, AudioDeckRefs } from "../components/NowPlaying";
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
  playbackWholeShare: number;
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

const AUDIO_DECKS: AudioDeck[] = ["primary", "secondary"];
const AUDIO_STEMS: AudioStem[] = ["original", "instrumental"];
const SYNC_DRIFT_SECONDS = 0.005;
const SYNC_INTERVAL_MS = 100;

export function useAudioPlayback({
  selected,
  playbackWholeShare,
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
  const primaryOriginalAudioRef = useRef<HTMLAudioElement | null>(null);
  const primaryInstrumentalAudioRef = useRef<HTMLAudioElement | null>(null);
  const secondaryOriginalAudioRef = useRef<HTMLAudioElement | null>(null);
  const secondaryInstrumentalAudioRef = useRef<HTMLAudioElement | null>(null);
  const audioRefs = useMemo<AudioDeckRefs>(() => ({
    primary: {
      original: primaryOriginalAudioRef,
      instrumental: primaryInstrumentalAudioRef,
    },
    secondary: {
      original: secondaryOriginalAudioRef,
      instrumental: secondaryInstrumentalAudioRef,
    },
  }), []);

  const [activeDeck, setActiveDeck] = useState<AudioDeck>("primary");
  const activeDeckRef = useRef<AudioDeck>("primary");
  const fadeFrameRef = useRef<number | null>(null);
  const fadeIntervalRef = useRef<number | null>(null);
  const fadeTimeoutRef = useRef<number | null>(null);
  const isCrossfadingRef = useRef(false);
  const playbackWholeShareRef = useRef(playbackWholeShare);
  const deckTracksRef = useRef<Record<AudioDeck, Track | null>>({
    primary: null,
    secondary: null,
  });
  const deckVolumeRef = useRef<Record<AudioDeck, number>>({
    primary: 1,
    secondary: 0,
  });
  const deckLoadIdRef = useRef<Record<AudioDeck, number>>({
    primary: 0,
    secondary: 0,
  });
  const isPlayingRef = useRef(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [currentDuration, setCurrentDuration] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);

  const setPlaybackState = useCallback((playing: boolean) => {
    isPlayingRef.current = playing;
    setIsPlaying(playing);
  }, []);

  const getStemAudio = useCallback((deck: AudioDeck, stem: AudioStem): HTMLAudioElement | null =>
    audioRefs[deck][stem].current, [audioRefs]);

  const getDeckAudios = useCallback((deck: AudioDeck): HTMLAudioElement[] =>
    AUDIO_STEMS.flatMap((stem) => {
      const audio = getStemAudio(deck, stem);
      return audio ? [audio] : [];
    }), [getStemAudio]);

  const getAllAudios = useCallback((): HTMLAudioElement[] =>
    AUDIO_DECKS.flatMap((deck) => getDeckAudios(deck)), [getDeckAudios]);

  const getAudio = useCallback((deck: AudioDeck): HTMLAudioElement | null => {
    const original = getStemAudio(deck, "original");
    if (original && hasAudioSource(original)) {
      return original;
    }
    return getStemAudio(deck, "instrumental");
  }, [getStemAudio]);

  const isDeckPlaying = useCallback((deck: AudioDeck): boolean => {
    const audio = getAudio(deck);
    return Boolean(audio && hasAudioSource(audio) && !audio.paused);
  }, [getAudio]);

  const areDeckAudiosPlaying = useCallback((deck: AudioDeck): boolean => {
    const audios = getDeckAudios(deck).filter(hasAudioSource);
    return audios.length > 0 && audios.every((audio) => !audio.paused);
  }, [getDeckAudios]);

  const applyDeckVolume = useCallback((deck: AudioDeck) => {
    const deckVolume = deckVolumeRef.current[deck];
    const hasInstrumental = Boolean(deckTracksRef.current[deck]?.available_stems.includes("instrumental"));
    const wholeShare = hasInstrumental ? clampUnit(playbackWholeShareRef.current) : 1;
    for (const stem of AUDIO_STEMS) {
      const audio = getStemAudio(deck, stem);
      if (audio) {
        const stemVolume = hasInstrumental
          ? (stem === "original" ? wholeShare : 1 - wholeShare)
          : (stem === "original" ? 1 : 0);
        audio.volume = deckVolume * stemVolume;
      }
    }
  }, [getStemAudio]);

  const applyAllDeckVolumes = useCallback(() => {
    for (const deck of AUDIO_DECKS) {
      applyDeckVolume(deck);
    }
  }, [applyDeckVolume]);

  const pauseDeck = useCallback((deck: AudioDeck) => {
    for (const audio of getDeckAudios(deck)) {
      audio.pause();
    }
  }, [getDeckAudios]);

  const playDeck = useCallback(async (deck: AudioDeck): Promise<boolean> => {
    const audios = getDeckAudios(deck).filter(hasAudioSource);
    if (audios.length === 0) {
      return false;
    }
    const reference = getAudio(deck);
    if (reference && reference.readyState >= 1) {
      syncAudiosToReference(reference, audios, 0);
    }
    await Promise.all(audios.map((audio) => audio.play().catch(() => undefined)));
    if (reference && reference.readyState >= 1) {
      syncAudiosToReference(reference, audios, 0);
      window.requestAnimationFrame(() => {
        syncAudiosToReference(reference, audios, SYNC_DRIFT_SECONDS);
      });
    }
    return isDeckPlaying(deck);
  }, [getAudio, getDeckAudios, isDeckPlaying]);

  const primeDeckAudios = useCallback((deck: AudioDeck) => {
    const audios = getDeckAudios(deck).filter(hasAudioSource);
    for (const audio of audios) {
      audio.load();
    }
  }, [getDeckAudios]);

  const activateDeck = useCallback((deck: AudioDeck) => {
    activeDeckRef.current = deck;
    setActiveDeck(deck);
    setPlaybackState(isDeckPlaying(deck));
  }, [isDeckPlaying, setPlaybackState]);

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
    for (const deck of AUDIO_DECKS) {
      pauseDeck(deck);
    }
    setPlaybackState(false);
  }, [cancelFade, pauseDeck, setPlaybackState]);

  const setDeckCurrentTime = useCallback((
    deck: AudioDeck,
    seconds: number,
    loadId?: number
  ) => {
    for (const audio of getDeckAudios(deck)) {
      if (!hasAudioSource(audio)) {
        continue;
      }
      setAudioCurrentTime(audio, seconds, () =>
        loadId === undefined || deckLoadIdRef.current[deck] === loadId
      );
    }
  }, [getDeckAudios]);

  const syncDeckToReference = useCallback((deck: AudioDeck) => {
    const reference = getAudio(deck);
    if (!reference || !hasAudioSource(reference) || reference.readyState < 1) {
      return;
    }
    syncAudiosToReference(reference, getDeckAudios(deck), SYNC_DRIFT_SECONDS);
  }, [getAudio, getDeckAudios]);

  const prepareDeckForTrack = useCallback((deck: AudioDeck, track: Track): number => {
    deckTracksRef.current[deck] = track;
    deckLoadIdRef.current[deck] += 1;
    const loadId = deckLoadIdRef.current[deck];
    const availableStems = new Set<AudioStem>(["original", ...track.available_stems]);

    for (const stem of AUDIO_STEMS) {
      const audio = getStemAudio(deck, stem);
      if (!audio) {
        continue;
      }
      audio.pause();
      if (!availableStems.has(stem)) {
        clearAudioSource(audio);
        continue;
      }

      const nextSrc = new URL(audioUrl(track.id, stem), window.location.href).href;
      if (audio.src !== nextSrc) {
        audio.src = nextSrc;
      }
    }

    applyDeckVolume(deck);
    primeDeckAudios(deck);
    return loadId;
  }, [applyDeckVolume, getStemAudio, primeDeckAudios]);

  const runWhenReferenceReady = useCallback((
    deck: AudioDeck,
    loadId: number,
    callback: (audio: HTMLAudioElement) => void
  ) => {
    const audio = getAudio(deck);
    if (!audio) {
      return;
    }
    const run = () => {
      if (deckLoadIdRef.current[deck] === loadId) {
        callback(audio);
      }
    };
    if (audio.readyState >= 1) {
      run();
      return;
    }
    audio.addEventListener("loadedmetadata", run, { once: true });
  }, [getAudio]);

  useEffect(() => {
    const syncPlaybackState = () => {
      setPlaybackState(isDeckPlaying(activeDeckRef.current));
    };
    const audios = getAllAudios();
    for (const audio of audios) {
      audio.addEventListener("play", syncPlaybackState);
      audio.addEventListener("pause", syncPlaybackState);
      audio.addEventListener("ended", syncPlaybackState);
    }
    syncPlaybackState();
    return () => {
      for (const audio of audios) {
        audio.removeEventListener("play", syncPlaybackState);
        audio.removeEventListener("pause", syncPlaybackState);
        audio.removeEventListener("ended", syncPlaybackState);
      }
    };
  }, [getAllAudios, isDeckPlaying, setPlaybackState]);

  useEffect(() => {
    const interval = window.setInterval(() => {
      for (const deck of AUDIO_DECKS) {
        syncDeckToReference(deck);
      }

      const audio = getAudio(activeDeckRef.current);
      if (!audio || audio.paused) {
        return;
      }
      setCurrentTime(audio.currentTime);
      if (Number.isFinite(audio.duration)) {
        setCurrentDuration(audio.duration);
      }
    }, SYNC_INTERVAL_MS);
    return () => window.clearInterval(interval);
  }, [getAudio, syncDeckToReference]);

  useEffect(() => {
    playbackWholeShareRef.current = playbackWholeShare;
    applyAllDeckVolumes();
    const audio = getAudio(activeDeckRef.current);
    if (audio && hasAudioSource(audio)) {
      setCurrentTime(audio.currentTime);
      if (Number.isFinite(audio.duration)) {
        setCurrentDuration(audio.duration);
      }
    }
    if (isPlayingRef.current) {
      syncDeckToReference(activeDeckRef.current);
      if (!areDeckAudiosPlaying(activeDeckRef.current)) {
        void playDeck(activeDeckRef.current).then(setPlaybackState);
      }
    }
  }, [
    applyAllDeckVolumes,
    areDeckAudiosPlaying,
    getAudio,
    playDeck,
    playbackWholeShare,
    setPlaybackState,
    syncDeckToReference,
  ]);

  const playTrack = useCallback((track: Track, startSeconds?: number) => {
    cancelFade();
    const deck = activeDeckRef.current;
    const inactiveDeck = getInactiveDeck(deck);

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

    pauseDeck(inactiveDeck);
    deckVolumeRef.current[deck] = 1;
    deckVolumeRef.current[inactiveDeck] = 0;
    applyDeckVolume(inactiveDeck);
    const loadId = prepareDeckForTrack(deck, track);

    runWhenReferenceReady(deck, loadId, (audio) => {
      const defaultStartSeconds = similarityMode === "track" && !trackAutoplay ? 39 : 0;
      const requestedStartSeconds = startSeconds ?? defaultStartSeconds;
      const duration = Number.isFinite(audio.duration) ? audio.duration : requestedStartSeconds;
      const nextCurrentTime = Math.min(requestedStartSeconds, Math.max(0, duration - 0.01));
      setDeckCurrentTime(deck, nextCurrentTime, loadId);
      setCurrentTime(nextCurrentTime);
      if (Number.isFinite(audio.duration)) {
        setCurrentDuration(audio.duration);
      }
      void playDeck(deck).then(setPlaybackState);
    });
  }, [
    applyDeckVolume,
    cancelFade,
    pauseDeck,
    playDeck,
    prepareDeckForTrack,
    resetAutoplayHistory,
    resetTrackAutoplayHistory,
    runWhenReferenceReady,
    setDeckCurrentTime,
    setPlaybackState,
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
    if (!fromAudio || !hasAudioSource(fromAudio)) {
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

    deckVolumeRef.current[fromDeck] = 1;
    deckVolumeRef.current[toDeck] = 0;
    const loadId = prepareDeckForTrack(toDeck, track);

    runWhenReferenceReady(toDeck, loadId, (toAudio) => {
      const duration = Number.isFinite(toAudio.duration) ? toAudio.duration : startSeconds;
      const nextCurrentTime = Math.min(startSeconds, Math.max(0, duration - 0.01));
      setDeckCurrentTime(toDeck, nextCurrentTime, loadId);
      void playDeck(toDeck)
        .then((toDeckPlaying) => {
          if (!toDeckPlaying) {
            isCrossfadingRef.current = false;
            playTrack(track, startSeconds);
            return;
          }

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

            pauseDeck(fromDeck);
            deckVolumeRef.current[fromDeck] = 0;
            deckVolumeRef.current[toDeck] = 1;
            applyDeckVolume(fromDeck);
            applyDeckVolume(toDeck);
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
            if (Number.isFinite(toAudio.duration)) {
              setCurrentDuration(toAudio.duration);
            }
            activateDeck(toDeck);
          };
          const updateFade = (now: number): boolean => {
            if (transitionCommitted) {
              return true;
            }
            const progress = Math.min(1, (now - fadeStart) / (CROSSFADE_SECONDS * 1000));
            const eased = progress * progress * (3 - 2 * progress);
            deckVolumeRef.current[fromDeck] = 1 - eased;
            deckVolumeRef.current[toDeck] = eased;
            applyDeckVolume(fromDeck);
            applyDeckVolume(toDeck);

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
        });
    });
  }, [
    activateDeck,
    applyDeckVolume,
    buildRoutePreviewFromHistory,
    cancelFade,
    getAudio,
    pauseDeck,
    playDeck,
    playTrack,
    prepareDeckForTrack,
    recordAutoplayVisit,
    runWhenReferenceReady,
    segmentAutoplay,
    setDeckCurrentTime,
    setRoutePreview,
    setSelected,
    setTrackAutoplayPreview,
    similarityMode,
    trackAutoplayPreviewKeyRef,
    tracksRef,
  ]);

  const handleTimeUpdate = useCallback((
    deck: AudioDeck,
    stem: AudioStem,
    event: SyntheticEvent<HTMLAudioElement>
  ) => {
    if (activeDeckRef.current !== deck || stem !== "original") {
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
    const deck = activeDeckRef.current;
    const audio = getAudio(deck);
    if (!audio || !hasAudioSource(audio)) {
      return;
    }
    if (audio.paused) {
      void playDeck(deck).then(setPlaybackState);
      return;
    }
    pauseDeck(deck);
    setPlaybackState(false);
  }, [getAudio, pauseDeck, playDeck, setPlaybackState]);

  const seekTo = useCallback((seconds: number) => {
    const deck = activeDeckRef.current;
    const audio = getAudio(deck);
    if (!audio) {
      return;
    }
    const duration = Number.isFinite(audio.duration) ? audio.duration : currentDuration;
    const clamped = Math.min(Math.max(0, seconds), Math.max(0, duration));
    setDeckCurrentTime(deck, clamped);
    setCurrentTime(clamped);
  }, [currentDuration, getAudio, setDeckCurrentTime]);

  return {
    activeDeck,
    activeDeckRef,
    currentTime,
    currentDuration,
    isPlaying,
    setCurrentTime,
    audioRefs,
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

function hasAudioSource(audio: HTMLAudioElement): boolean {
  return Boolean(audio.currentSrc || audio.src);
}

function clampUnit(value: number): number {
  if (!Number.isFinite(value)) {
    return 1;
  }
  return Math.min(1, Math.max(0, value));
}

function syncAudiosToReference(
  reference: HTMLAudioElement,
  audios: HTMLAudioElement[],
  toleranceSeconds: number
): void {
  for (const audio of audios) {
    if (
      audio === reference ||
      !hasAudioSource(audio) ||
      audio.readyState < 1 ||
      Math.abs(audio.currentTime - reference.currentTime) <= toleranceSeconds
    ) {
      continue;
    }
    try {
      audio.currentTime = reference.currentTime;
    } catch {
      // Ignore transient seek failures while a hidden stem is still buffering.
    }
  }
}

function clearAudioSource(audio: HTMLAudioElement): void {
  audio.pause();
  audio.removeAttribute("src");
  audio.load();
}

function setAudioCurrentTime(
  audio: HTMLAudioElement,
  seconds: number,
  isCurrent: () => boolean
): void {
  const apply = () => {
    if (!isCurrent()) {
      return;
    }
    const duration = Number.isFinite(audio.duration) ? audio.duration : seconds;
    const target = Math.min(seconds, Math.max(0, duration - 0.01));
    try {
      audio.currentTime = target;
    } catch {
      // The browser can reject a seek while metadata is still settling.
    }
  };

  if (audio.readyState >= 1) {
    apply();
    return;
  }
  audio.addEventListener("loadedmetadata", apply, { once: true });
}
