export type TrackStatus = "queued" | "processing" | "ready" | "error";
export type FeedbackLabel = "similar" | "not_similar";
export type ViewMode = "2d" | "3d";
export type SimilarityMode = "track" | "segment";
export type AudioStem = "original" | "vocals" | "instrumental";

export type SimilarTrack = {
  id: string;
  score: number;
};

export type SimilarSegment = SimilarTrack & {
  segment_index: number;
  start_seconds: number;
};

export type SimilarityMix = {
  whole: number;
  vocals: number;
  instrumental: number;
  style: number;
  cover: number;
};

export type Track = {
  id: string;
  filename: string;
  title: string;
  artist: string | null;
  album: string | null;
  status: TrackStatus;
  error: string | null;
  x: number | null;
  y: number | null;
  z: number | null;
  cluster: number | null;
  segment_count?: number;
  available_stems: AudioStem[];
  similar: SimilarTrack[];
};
