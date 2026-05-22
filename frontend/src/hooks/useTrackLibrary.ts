import { useCallback, useEffect, useMemo, useState } from "react";
import {
  deleteAllTracks,
  deleteTrack,
  fetchTracks,
  uploadTracks
} from "../api";
import type { Track } from "../types";

type UseTrackLibraryOptions = {
  onTracksLoaded: (tracks: Track[]) => void;
};

export function useTrackLibrary({ onTracksLoaded }: UseTrackLibraryOptions) {
  const [tracks, setTracks] = useState<Track[]>([]);
  const [message, setMessage] = useState("");
  const [dismissedErrorIds, setDismissedErrorIds] = useState<Set<string>>(() => new Set());

  const loadTracks = useCallback(async () => {
    const nextTracks = await fetchTracks();
    setTracks(nextTracks);
    onTracksLoaded(nextTracks);
  }, [onTracksLoaded]);

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

  const visibleFailedTracks = useMemo(() => {
    const failedTracks = tracks.filter((track) => track.status === "error");
    return failedTracks.filter((track) => !dismissedErrorIds.has(track.id));
  }, [dismissedErrorIds, tracks]);

  const handleFiles = useCallback(
    async (files: FileList | File[]) => {
      const selectedFiles = Array.from(files).filter(isAudioFile);
      if (selectedFiles.length === 0) {
        setMessage("No audio files found");
        return;
      }
      setMessage("Uploading");
      await uploadTracks(selectedFiles);
      setMessage("");
      await loadTracks();
    },
    [loadTracks]
  );

  const deleteTrackFromLibrary = useCallback(
    async (track: Track) => {
      await deleteTrack(track.id);
      setDismissedErrorIds((current) => {
        const next = new Set(current);
        next.delete(track.id);
        return next;
      });
      await loadTracks();
    },
    [loadTracks]
  );

  const deleteAllTracksFromLibrary = useCallback(async () => {
    await deleteAllTracks();
    setDismissedErrorIds(new Set());
    await loadTracks();
  }, [loadTracks]);

  const dismissTrackError = useCallback((track: Track) => {
    setDismissedErrorIds((current) => {
      const next = new Set(current);
      next.add(track.id);
      return next;
    });
  }, []);

  return {
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
  };
}

function isAudioFile(file: File): boolean {
  if (file.type.startsWith("audio/")) {
    return true;
  }
  return /\.(aac|flac|m4a|mp3|ogg|wav)$/i.test(file.name);
}
