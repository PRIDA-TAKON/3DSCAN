-- Create jobs table for tracking 3DGS tasks
CREATE TABLE IF NOT EXISTS public.jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT timezone('utc'::text, now()) NOT NULL,
    video_url TEXT NOT NULL,
    status TEXT DEFAULT 'PENDING' CHECK (status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED')),
    message TEXT,
    result_url TEXT,
    user_id UUID REFERENCES auth.users(id) -- Optional: if you want auth
);

-- Enable Row Level Security (RLS)
ALTER TABLE public.jobs ENABLE ROW LEVEL SECURITY;

-- Create policies (Example: Allow all for now, or restrict by user)
CREATE POLICY "Allow authenticated full access" ON public.jobs
    FOR ALL USING (true);

-- Create storage bucket for medical videos
-- Note: You should create a bucket named '3d-scans' in Supabase Dashbord and set it to Public or Managed RLS.
