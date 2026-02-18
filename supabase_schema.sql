-- Create Enum for Job Status
CREATE TYPE job_status AS ENUM (
    'PENDING',
    'SFM_QUEUED',
    'SFM_RUNNING',
    'SFM_COMPLETED',
    'SFM_FAILED',
    'TRAINING_QUEUED',
    'TRAINING_RUNNING',
    'TRAINING_COMPLETED',
    'TRAINING_FAILED',
    'CONVERSION_QUEUED',
    'CONVERSION_RUNNING',
    'COMPLETED',
    'FAILED'
);

-- Create jobs table for tracking 3DGS tasks
CREATE TABLE IF NOT EXISTS public.jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    
    -- Inputs
    video_url TEXT NOT NULL,
    config JSONB DEFAULT '{}'::jsonb, -- Hyperparameters
    
    -- Status tracking
    status job_status DEFAULT 'PENDING',
    message TEXT,
    
    -- Google Drive Links/IDs
    drive_folder_id TEXT, -- Main folder for this job
    sfm_url TEXT,         -- Link to sparse_model.zip
    model_url TEXT,       -- Link to model.ply / output.zip
    result_url TEXT,      -- Link to .splat file (Final Result)
    
    -- Logs & Metadata
    kaggle_kernel_run_ids JSONB DEFAULT '{}'::jsonb, -- { "sfm": "run_id", "train": "run_id" }
    logs JSONB DEFAULT '[]'::jsonb,
    
    user_id UUID REFERENCES auth.users(id)
);

-- Trigger to update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_jobs_updated_at
    BEFORE UPDATE ON public.jobs
    FOR EACH ROW
    EXECUTE PROCEDURE update_updated_at_column();

-- Enable RLS
ALTER TABLE public.jobs ENABLE ROW LEVEL SECURITY;

-- Policies
CREATE POLICY "Enable read access for all users" ON public.jobs FOR SELECT USING (true);
CREATE POLICY "Enable insert for authenticated users only" ON public.jobs FOR INSERT WITH CHECK (auth.role() = 'authenticated');
CREATE POLICY "Enable update for service role/dispatchers" ON public.jobs FOR UPDATE USING (true); -- Simplify for MVP, tighten later
