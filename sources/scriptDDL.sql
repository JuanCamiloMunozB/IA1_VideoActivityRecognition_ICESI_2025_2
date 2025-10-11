CREATE TABLE videos (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    fps NUMERIC,
    resolution TEXT,
    width INT,
    height INT,
    duration_sec NUMERIC,
    lighting NUMERIC,                  -- brillo promedio
    upload_date TIMESTAMP DEFAULT now()
);

CREATE TABLE landmarks (
    id BIGSERIAL PRIMARY KEY,
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    landmarks JSONB,                  -- Ej: {"cadera": [x,y], "rodilla": [x,y], ...}
    created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE annotations (
    id BIGSERIAL PRIMARY KEY,
    video_id UUID REFERENCES videos(id) ON DELETE CASCADE,
    label TEXT NOT NULL,                -- "sentado", "saltando", "inclinación lateral", etc.
    start_time_sec NUMERIC,
    end_time_sec NUMERIC,
    created_at TIMESTAMP DEFAULT now()
);
