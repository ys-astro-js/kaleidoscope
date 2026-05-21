export type TrackStatus = "queued" | "processing" | "ready" | "error";

export type Track = {
  id: string;
  filename: string;
  title: string;
  artist: string | null;
  status: TrackStatus;
  error: string | null;
  x: number | null;
  y: number | null;
  z: number | null;
  similar: string[];
};

