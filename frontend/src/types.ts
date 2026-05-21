export type TrackStatus = "queued" | "processing" | "ready" | "error";
export type FeedbackLabel = "similar" | "not_similar";
export type ViewMode = "2d" | "3d";

export type SimilarTrack = {
  id: string;
  score: number;
};

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
  cluster: number | null;
  similar: SimilarTrack[];
};
