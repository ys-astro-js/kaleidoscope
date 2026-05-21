import type { FeedbackLabel, Track } from "./types";

export async function fetchTracks(): Promise<Track[]> {
  const response = await fetch("/api/tracks");
  if (!response.ok) {
    throw new Error("Failed to fetch tracks");
  }
  return response.json();
}

export async function uploadTracks(files: File[]): Promise<void> {
  await Promise.all(
    files.map(async (file) => {
      const form = new FormData();
      form.append("upload", file);
      const response = await fetch("/api/tracks", {
        method: "POST",
        body: form
      });
      if (!response.ok) {
        throw new Error(`Failed to upload ${file.name}`);
      }
    })
  );
}

export async function deleteTrack(trackId: string): Promise<void> {
  const response = await fetch(`/api/tracks/${trackId}`, {
    method: "DELETE"
  });
  if (!response.ok) {
    throw new Error("Failed to delete track");
  }
}

export async function deleteAllTracks(): Promise<void> {
  const response = await fetch("/api/tracks?confirm=true", {
    method: "DELETE"
  });
  if (!response.ok) {
    throw new Error("Failed to delete tracks");
  }
}

export async function submitFeedback(
  queryTrackId: string,
  candidateTrackId: string,
  label: FeedbackLabel
): Promise<void> {
  const response = await fetch("/api/feedback", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      query_track_id: queryTrackId,
      candidate_track_id: candidateTrackId,
      label
    })
  });
  if (!response.ok) {
    throw new Error("Failed to submit feedback");
  }
}

export function audioUrl(trackId: string): string {
  return `/api/tracks/${trackId}/audio`;
}

export function artUrl(trackId: string): string {
  return `/api/tracks/${trackId}/art`;
}
